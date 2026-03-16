"""Tests for the navigator module: planner, autopilot, and waypoints."""

from __future__ import annotations

import pytest

from vimana.models import (
    ActionType,
    InfrastructureState,
    Region,
    Resource,
    ResourceSpec,
    ResourceStatus,
    ResourceType,
    SimulationScenario,
    Waypoint,
)
from vimana.navigator.planner import FlightPlanner, _heuristic_plan
from vimana.navigator.waypoints import (
    WAYPOINT_MINIMAL,
    WAYPOINT_STANDARD,
    WAYPOINT_HIGH_AVAILABILITY,
    WaypointGraph,
    state_to_waypoint,
)
from vimana.simulation import CloudSimulator


# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------

class TestWaypoint:
    def test_distance_same(self):
        wp = WAYPOINT_MINIMAL
        assert wp.distance_to(wp) == 0.0

    def test_distance_different(self):
        d = WAYPOINT_MINIMAL.distance_to(WAYPOINT_HIGH_AVAILABILITY)
        assert d > 0

    def test_distance_asymmetric_regions(self):
        a = Waypoint(name="a", compute_vcpus=4, memory_gb=8, regions=[Region.US_EAST_1])
        b = Waypoint(name="b", compute_vcpus=4, memory_gb=8, regions=[Region.US_EAST_1, Region.EU_WEST_1])
        assert a.distance_to(b) > 0


class TestWaypointGraph:
    def test_default_graph_has_waypoints(self):
        graph = WaypointGraph.default()
        assert len(graph.waypoints) >= 5
        assert "minimal" in graph.waypoints
        assert "standard" in graph.waypoints

    def test_calculate_route_same_node(self):
        graph = WaypointGraph.default()
        route = graph.calculate_route(WAYPOINT_MINIMAL, WAYPOINT_MINIMAL)
        assert len(route) == 1  # just the target

    def test_calculate_route_connected(self):
        graph = WaypointGraph.default()
        route = graph.calculate_route(WAYPOINT_MINIMAL, WAYPOINT_STANDARD)
        assert len(route) >= 1
        assert route[-1].name == "standard"

    def test_calculate_route_multi_hop(self):
        graph = WaypointGraph.default()
        route = graph.calculate_route(WAYPOINT_MINIMAL, WAYPOINT_HIGH_AVAILABILITY)
        assert len(route) >= 1

    def test_add_custom_waypoint(self):
        graph = WaypointGraph.default()
        custom = Waypoint(name="custom", compute_vcpus=12, memory_gb=24)
        graph.add_waypoint(custom)
        assert "custom" in graph.waypoints


class TestStateToWaypoint:
    def test_empty_state(self):
        state = InfrastructureState()
        wp = state_to_waypoint(state)
        assert wp.compute_vcpus == 0
        assert wp.memory_gb == 0.0

    def test_state_with_resources(self):
        spec = ResourceSpec(
            resource_type=ResourceType.VM,
            vcpus=4,
            memory_gb=8.0,
            region=Region.US_EAST_1,
            replicas=2,
        )
        resource = Resource(spec=spec, status=ResourceStatus.RUNNING, cost_per_hour=0.5)
        state = InfrastructureState(resources=[resource])
        state.refresh_totals()
        wp = state_to_waypoint(state)
        assert wp.compute_vcpus == 8  # 4 vcpus * 2 replicas
        assert wp.memory_gb == 16.0


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestFlightPlanner:
    def _make_empty_state(self) -> InfrastructureState:
        return InfrastructureState()

    def _make_running_state(self) -> InfrastructureState:
        spec = ResourceSpec(
            resource_type=ResourceType.VM,
            vcpus=4,
            memory_gb=8.0,
            region=Region.US_EAST_1,
        )
        resource = Resource(spec=spec, status=ResourceStatus.RUNNING, cost_per_hour=0.5)
        state = InfrastructureState(resources=[resource])
        state.refresh_totals()
        return state

    def test_plan_from_empty(self):
        planner = FlightPlanner(use_llm=False)
        plan = planner.plan(self._make_empty_state(), WAYPOINT_STANDARD)
        assert len(plan.actions) > 0
        provision_actions = [a for a in plan.actions if a.action_type == ActionType.PROVISION]
        assert len(provision_actions) > 0

    def test_plan_already_at_goal(self):
        state = self._make_running_state()
        tiny_goal = Waypoint(name="tiny", compute_vcpus=2, memory_gb=4.0)
        planner = FlightPlanner(use_llm=False)
        plan = planner.plan(state, tiny_goal)
        # Should suggest termination or no actions since we exceed the goal
        assert plan is not None

    def test_plan_heals_degraded(self):
        spec = ResourceSpec(resource_type=ResourceType.VM, vcpus=4, memory_gb=8.0)
        resource = Resource(spec=spec, status=ResourceStatus.DEGRADED, cost_per_hour=0.5)
        state = InfrastructureState(resources=[resource])
        state.refresh_totals()

        planner = FlightPlanner(use_llm=False)
        plan = planner.plan(state, WAYPOINT_STANDARD)
        heal_actions = [a for a in plan.actions if a.action_type == ActionType.HEAL]
        assert len(heal_actions) >= 1

    def test_plan_route(self):
        planner = FlightPlanner(use_llm=False)
        plans = planner.plan_route(self._make_empty_state(), WAYPOINT_HIGH_AVAILABILITY)
        assert len(plans) >= 1


# ---------------------------------------------------------------------------
# Autopilot (basic, no full simulation needed)
# ---------------------------------------------------------------------------

class TestAutopilot:
    def test_dry_run(self):
        from vimana.cloud.health import HealthMonitor, SelfHealer
        from vimana.cloud.migrator import CloudMigrator
        from vimana.cloud.provisioner import CloudProvisioner
        from vimana.navigator.autopilot import Autopilot

        scenario = SimulationScenario(
            name="test",
            total_ticks=10,
            initial_resources=[
                ResourceSpec(resource_type=ResourceType.VM, vcpus=2, memory_gb=4.0)
            ],
        )
        sim = CloudSimulator(scenario=scenario)
        sim.initialize()

        provisioner = CloudProvisioner(simulator=sim)
        migrator = CloudMigrator(simulator=sim)
        hm = HealthMonitor(simulator=sim)
        sh = SelfHealer(simulator=sim)

        autopilot = Autopilot(
            simulator=sim,
            provisioner=provisioner,
            migrator=migrator,
            health_monitor=hm,
            self_healer=sh,
            dry_run=True,
            verbose=False,
        )

        planner = FlightPlanner(use_llm=False)
        plan = planner.plan(sim.state, WAYPOINT_STANDARD)
        log = autopilot.fly(plan)

        assert log.dry_run is True
        assert all(e.result.success for e in log.entries)
