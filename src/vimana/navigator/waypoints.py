"""Infrastructure waypoints and navigation graph.

A Waypoint defines a desired infrastructure state. A WaypointGraph enables
pathfinding between states, so the AI can plan a route from its current
configuration to a target configuration.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from vimana.models import (
    InfrastructureState,
    Region,
    ResourceType,
    Waypoint,
)


def state_to_waypoint(state: InfrastructureState, name: str = "current") -> Waypoint:
    """Convert a live infrastructure state into a Waypoint."""
    return Waypoint(
        name=name,
        compute_vcpus=state.total_vcpus,
        memory_gb=state.total_memory_gb,
        replicas=sum(r.spec.replicas for r in state.resources),
        regions=state.regions_active or [Region.US_EAST_1],
        max_cost_per_hour=state.total_cost_per_hour * 1.5,
        resource_types=list({r.spec.resource_type for r in state.resources})
        or [ResourceType.VM],
    )


# ---------------------------------------------------------------------------
# Predefined waypoints (common infrastructure profiles)
# ---------------------------------------------------------------------------

WAYPOINT_MINIMAL = Waypoint(
    name="minimal",
    compute_vcpus=2,
    memory_gb=4.0,
    replicas=1,
    regions=[Region.US_EAST_1],
    max_cost_per_hour=0.50,
    resource_types=[ResourceType.CONTAINER],
)

WAYPOINT_STANDARD = Waypoint(
    name="standard",
    compute_vcpus=8,
    memory_gb=16.0,
    replicas=3,
    regions=[Region.US_EAST_1],
    max_cost_per_hour=3.00,
    resource_types=[ResourceType.VM],
)

WAYPOINT_HIGH_AVAILABILITY = Waypoint(
    name="high-availability",
    compute_vcpus=16,
    memory_gb=64.0,
    replicas=6,
    regions=[Region.US_EAST_1, Region.US_WEST_2, Region.EU_WEST_1],
    max_cost_per_hour=15.00,
    min_availability=0.999,
    resource_types=[ResourceType.VM, ResourceType.LOAD_BALANCER],
)

WAYPOINT_BURST = Waypoint(
    name="burst",
    compute_vcpus=32,
    memory_gb=128.0,
    replicas=10,
    regions=[Region.US_EAST_1, Region.US_WEST_2],
    max_cost_per_hour=25.00,
    resource_types=[ResourceType.VM, ResourceType.CONTAINER],
)

WAYPOINT_COST_OPTIMIZED = Waypoint(
    name="cost-optimized",
    compute_vcpus=4,
    memory_gb=8.0,
    replicas=2,
    regions=[Region.US_EAST_1],
    max_cost_per_hour=1.00,
    resource_types=[ResourceType.SERVERLESS],
)


# ---------------------------------------------------------------------------
# WaypointGraph
# ---------------------------------------------------------------------------

@dataclass
class WaypointGraph:
    """Navigation graph of infrastructure waypoints.

    Provides pathfinding (A*) so the planner can determine the sequence of
    intermediate states to traverse when moving from current to target
    infrastructure.
    """

    waypoints: dict[str, Waypoint] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> WaypointGraph:
        """Build a default graph with standard waypoints."""
        graph = cls()
        for wp in [
            WAYPOINT_MINIMAL,
            WAYPOINT_STANDARD,
            WAYPOINT_HIGH_AVAILABILITY,
            WAYPOINT_BURST,
            WAYPOINT_COST_OPTIMIZED,
        ]:
            graph.add_waypoint(wp)

        # Define connectivity
        graph.connect("minimal", "standard")
        graph.connect("minimal", "cost-optimized")
        graph.connect("standard", "high-availability")
        graph.connect("standard", "burst")
        graph.connect("standard", "cost-optimized")
        graph.connect("high-availability", "burst")
        graph.connect("cost-optimized", "standard")
        return graph

    def add_waypoint(self, wp: Waypoint) -> None:
        self.waypoints[wp.name] = wp
        if wp.name not in self.edges:
            self.edges[wp.name] = []

    def connect(self, a: str, b: str, bidirectional: bool = True) -> None:
        if a in self.waypoints and b in self.waypoints:
            if b not in self.edges.get(a, []):
                self.edges.setdefault(a, []).append(b)
            if bidirectional and a not in self.edges.get(b, []):
                self.edges.setdefault(b, []).append(a)

    def calculate_route(
        self,
        current: Waypoint,
        target: Waypoint,
    ) -> list[Waypoint]:
        """Find the shortest route from *current* to *target* using A*.

        If neither endpoint is in the graph, the method inserts them
        temporarily and connects them to the nearest existing waypoint.
        """
        # Ensure endpoints are in the graph
        start_name = self._ensure_in_graph(current)
        end_name = self._ensure_in_graph(target)

        if start_name == end_name:
            return [target]

        # A* search
        open_set: list[tuple[float, str]] = [(0.0, start_name)]
        came_from: dict[str, str] = {}
        g_score: dict[str, float] = {start_name: 0.0}

        while open_set:
            _, current_name = heapq.heappop(open_set)
            if current_name == end_name:
                return self._reconstruct(came_from, end_name)

            for neighbor_name in self.edges.get(current_name, []):
                neighbor_wp = self.waypoints[neighbor_name]
                current_wp = self.waypoints[current_name]
                edge_cost = current_wp.distance_to(neighbor_wp)
                tentative_g = g_score[current_name] + edge_cost

                if tentative_g < g_score.get(neighbor_name, float("inf")):
                    came_from[neighbor_name] = current_name
                    g_score[neighbor_name] = tentative_g
                    h = neighbor_wp.distance_to(self.waypoints[end_name])
                    heapq.heappush(open_set, (tentative_g + h, neighbor_name))

        # No path found; direct jump
        return [target]

    # -- internal -----------------------------------------------------------

    def _ensure_in_graph(self, wp: Waypoint) -> str:
        """Add a waypoint to the graph if not already present, connecting to nearest."""
        if wp.name in self.waypoints:
            return wp.name

        self.add_waypoint(wp)

        # Connect to the two nearest existing waypoints
        dists = [
            (wp.distance_to(existing), name)
            for name, existing in self.waypoints.items()
            if name != wp.name
        ]
        dists.sort()
        for _, name in dists[:2]:
            self.connect(wp.name, name)

        return wp.name

    def _reconstruct(self, came_from: dict[str, str], end: str) -> list[Waypoint]:
        path: list[str] = [end]
        while end in came_from:
            end = came_from[end]
            path.append(end)
        path.reverse()
        # Skip the starting waypoint (the caller already knows where they are)
        return [self.waypoints[n] for n in path[1:]]
