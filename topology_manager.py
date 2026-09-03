import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

log = logging.getLogger(__name__)


# ── Topology communication graphs ──────────────────────────
# Each graph defines the directed edges of the topology.
# Format: list of (sender_role, receiver_role) tuples.
# "ALL" denotes a broadcast to all agents in the network.

TOPOLOGY_GRAPHS: Dict[str, List[Tuple[str, str]]] = {

    "linear": [
        # Strict sequential chain — no shortcuts
        ("SensorAgent",    "MonitorAgent"),
        ("MonitorAgent",   "SchedulerAgent"),
        ("SchedulerAgent", "ActuatorAgent"),
        ("ActuatorAgent",  "SupervisorAgent"),
    ],

    "star": [
        # All agents report to hub; hub redistributes
        ("SensorAgent",     "SupervisorAgent"),
        ("SupervisorAgent", "MonitorAgent"),
        ("MonitorAgent",    "SupervisorAgent"),
        ("SupervisorAgent", "SchedulerAgent"),
        ("SchedulerAgent",  "SupervisorAgent"),
        ("SupervisorAgent", "ActuatorAgent"),
    ],

    "ring": [
        # Circular chain — last agent wraps to first
        ("SensorAgent",     "MonitorAgent"),
        ("MonitorAgent",    "SchedulerAgent"),
        ("SchedulerAgent",  "ActuatorAgent"),
        ("ActuatorAgent",   "SupervisorAgent"),
        ("SupervisorAgent", "SensorAgent"),
    ],

    "tree": [
        # Root broadcasts; branches do not cross-communicate
        ("SensorAgent",    "ALL"),
        ("MonitorAgent",   "SchedulerAgent"),
        ("SchedulerAgent", "SupervisorAgent"),
        ("SchedulerAgent", "ActuatorAgent"),
    ],

    "mesh": [
        # Fully connected — every agent broadcasts
        ("SensorAgent",     "ALL"),
        ("MonitorAgent",    "ALL"),
        ("SchedulerAgent",  "ALL"),
        ("ActuatorAgent",   "ALL"),
        ("SupervisorAgent", "ALL"),
    ],
}

# ── Agent execution order per topology ─────────────────────
# Defines the sequence in which agents are called.
# Star topology calls SupervisorAgent multiple times
# because it acts as the routing hub between agents.
# SupervisorAgent should only terminate on its LAST appearance.

TOPOLOGY_ORDER: Dict[str, List[str]] = {
    "linear": [
        "SensorAgent", "MonitorAgent", "SchedulerAgent",
        "ActuatorAgent", "SupervisorAgent",
    ],
    "star": [
        "SensorAgent",
        "SupervisorAgent",   # hub receives from SensorAgent
        "MonitorAgent",
        "SupervisorAgent",   # hub receives from MonitorAgent
        "SchedulerAgent",
        "SupervisorAgent",   # hub receives from SchedulerAgent
        "ActuatorAgent",
        "SupervisorAgent",   # final hub validation (last appearance)
    ],
    "ring": [
        "SensorAgent", "MonitorAgent", "SchedulerAgent",
        "ActuatorAgent", "SupervisorAgent",
    ],
    "tree": [
        "SensorAgent", "MonitorAgent", "SchedulerAgent",
        "ActuatorAgent", "SupervisorAgent",
    ],
    "mesh": [
        "SensorAgent", "MonitorAgent", "SchedulerAgent",
        "ActuatorAgent", "SupervisorAgent",
    ],
}

# ── Decision keywords ──────────────────────────────────────
DECISION_KEYWORDS = [
    "EMERGENCY_STOP", "MAINTENANCE",
    "REDUCE_SPEED", "CONTINUE",
    "UNSAFE", "SAFE", "TERMINATE",
]


# ── Data structures ────────────────────────────────────────

@dataclass
class AgentMessage:
    """One message sent between agents in a topology run"""
    sender:   str
    receiver: str    # agent name or "ALL"
    content:  str
    turn:     int


@dataclass
class TopologyRunResult:
    """
    Complete result of one topology conversation.
    Contains all exchanged messages and extracted decisions.
    """
    topology:  str
    messages:  List[AgentMessage] = field(default_factory=list)
    decisions: Dict[str, str]     = field(default_factory=dict)

    def add_message(self, msg: AgentMessage) -> None:
        self.messages.append(msg)

    def get_context_for_agent(
        self,
        agent_name: str,
        topology: str
    ) -> str:
        """
        Build the conversation context visible to an agent
        based on the topology's communication constraints.

        This is the core of structural topology enforcement:
        agents only see what their topology allows.
        """
        if topology == "mesh":
            # Fully connected: agent sees entire history
            return "\n".join(
                f"[{m.sender}]: {m.content}"
                for m in self.messages
            )

        elif topology in ("linear", "ring"):
            # Agent sees only its direct predecessor's message
            relevant = [
                m for m in self.messages
                if m.receiver == agent_name
                or m.receiver == "ALL"
            ]
            if relevant:
                last = relevant[-1]
                return f"[{last.sender}]: {last.content}"
            return ""

        elif topology == "star":
            # Agent sees only messages from the hub or
            # messages addressed directly to it
            hub_messages = [
                m for m in self.messages
                if m.sender == "SupervisorAgent"
                or m.receiver == agent_name
            ]
            return "\n".join(
                f"[{m.sender}]: {m.content}"
                for m in hub_messages
            )

        elif topology == "tree":
            # Agent sees broadcast messages and
            # messages addressed to it directly
            relevant = [
                m for m in self.messages
                if m.receiver == "ALL"
                or m.receiver == agent_name
            ]
            return "\n".join(
                f"[{m.sender}]: {m.content}"
                for m in relevant
            )

        return ""


# ── Topology runner ────────────────────────────────────────

class TopologyRunner:
    """
    Executes a multi-agent conversation following a
    structurally correct network topology.

    Key difference from RoundRobinGroupChat:
        Each agent receives only the messages permitted by
        the topology's communication graph, not the full
        conversation history. This enforces the structural
        properties that distinguish topologies experimentally.

    Star topology fix:
        SupervisorAgent appears multiple times as a hub router.
        TERMINATE is only honoured on its last appearance to
        ensure SchedulerAgent and ActuatorAgent run before
        final validation.

    Usage:
        runner = TopologyRunner("mesh", agents_dict)
        result = await runner.run(initial_task)
        decisions = result.decisions
    """

    def __init__(
        self,
        topology_name: str,
        agents: Dict[str, AssistantAgent],
    ):
        """
        Args:
            topology_name : one of linear/star/ring/tree/mesh
            agents        : dict of {agent_name: AssistantAgent}
        """
        if topology_name not in TOPOLOGY_GRAPHS:
            raise ValueError(
                f"Unknown topology '{topology_name}'. "
                f"Valid: {sorted(TOPOLOGY_GRAPHS.keys())}"
            )

        self.topology  = topology_name
        self.agents    = agents
        self._result   = TopologyRunResult(topology=topology_name)

        # Count total SupervisorAgent appearances for star fix
        order = TOPOLOGY_ORDER[topology_name]
        self._total_supervisor_calls = order.count(
            "SupervisorAgent"
        )
        self._supervisor_call_count  = 0

    async def run(self, initial_task: str) -> TopologyRunResult:
        """
        Execute the full topology conversation.

        Args:
            initial_task : prompt sent to SensorAgent first

        Returns:
            TopologyRunResult with all messages and decisions
        """
        log.debug("Starting topology run: %s", self.topology)

        # Reset supervisor call counter for this run
        self._supervisor_call_count = 0

        # Step 1: SensorAgent reads the initial task
        sensor_response = await self._call_agent(
            agent_name="SensorAgent",
            context=initial_task,
        )
        self._record_message(
            sender="SensorAgent",
            receiver=self._next_receiver("SensorAgent"),
            content=sensor_response,
            turn=1,
        )

        # Step 2: Remaining agents in topology order
        order = TOPOLOGY_ORDER[self.topology]

        for idx, agent_name in enumerate(order[1:], start=2):
            context = self._build_context(
                agent_name=agent_name,
                initial_task=initial_task,
            )

            response = await self._call_agent(
                agent_name=agent_name,
                context=context,
            )

            self._record_message(
                sender=agent_name,
                receiver=self._next_receiver(agent_name),
                content=response,
                turn=idx,
            )

            log.debug(
                "Turn %d | %-18s -> %-18s | decision: %s",
                idx, agent_name,
                self._next_receiver(agent_name),
                self._result.decisions.get(agent_name, "none"),
            )

            # Terminate only when SupervisorAgent has made its
            # final appearance (fixes star topology UNKNOWN bug)
            if agent_name == "SupervisorAgent":
                self._supervisor_call_count += 1
                is_last_supervisor = (
                    self._supervisor_call_count
                    >= self._total_supervisor_calls
                )
                if ("TERMINATE" in response.upper()
                        and is_last_supervisor):
                    log.debug(
                        "SupervisorAgent final call — terminating."
                    )
                    break

        return self._result

    def print_conversation(self) -> None:
        """Print routing summary for debugging"""
        log.info(
            "Topology %s | %d messages",
            self.topology,
            len(self._result.messages),
        )
        for msg in self._result.messages:
            decision = self._result.decisions.get(
                msg.sender, "-"
            )
            log.info(
                "  Turn %2d | %-18s -> %-18s | %s",
                msg.turn, msg.sender, msg.receiver, decision,
            )

    # ── Private helpers ────────────────────────────────────

    async def _call_agent(
        self,
        agent_name: str,
        context: str,
    ) -> str:
        """
        Call one agent and return its text response.
        Returns an error string on failure to allow
        the run to continue rather than crashing.
        """
        agent = self.agents.get(agent_name)
        if agent is None:
            log.warning("Agent not found: %s", agent_name)
            return f"[{agent_name} not found]"

        try:
            token    = CancellationToken()
            response = await agent.on_messages(
                [TextMessage(content=context, source="user")],
                token,
            )
            if response and response.chat_message:
                content = response.chat_message.content
                self._extract_decision(agent_name, content)
                return content
        except Exception as exc:
            log.warning(
                "Agent %s raised exception: %s",
                agent_name, exc,
            )
            return f"[{agent_name} error: {str(exc)[:80]}]"

        return f"[{agent_name} no response]"

    def _build_context(
        self,
        agent_name: str,
        initial_task: str,
    ) -> str:
        """
        Build topology-constrained context for an agent.
        If no prior messages exist, fall back to initial task.
        """
        prior_context = self._result.get_context_for_agent(
            agent_name=agent_name,
            topology=self.topology,
        )

        if not prior_context:
            return initial_task

        topology_descriptions = {
            "mesh":   "Full conversation history (mesh topology):",
            "star":   "Message from hub agent (star topology):",
            "linear": "Message from predecessor (linear topology):",
            "ring":   "Message from predecessor (ring topology):",
            "tree":   "Broadcast message received (tree topology):",
        }

        header = topology_descriptions.get(
            self.topology,
            "Message received:"
        )

        return (
            f"{header}\n"
            f"{prior_context}\n\n"
            f"Respond as {agent_name}."
        )

    def _record_message(
        self,
        sender: str,
        receiver: str,
        content: str,
        turn: int,
    ) -> None:
        """Record a message in the run result"""
        self._result.add_message(AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            turn=turn,
        ))

    def _extract_decision(
        self,
        agent_name: str,
        content: str,
    ) -> None:
        """Extract and store decision keyword from response"""
        upper = content.upper()
        for keyword in DECISION_KEYWORDS:
            if keyword in upper:
                self._result.decisions[agent_name] = keyword
                break

    def _next_receiver(self, sender: str) -> str:
        """
        Return the intended receiver for a sender's message
        based on the topology graph.
        """
        graph = TOPOLOGY_GRAPHS[self.topology]
        for s, r in graph:
            if s == sender:
                return r
        return "ALL"