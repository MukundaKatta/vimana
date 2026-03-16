"""Data models for the Vimana self-orchestrating AI framework.

All core types used across the system: infrastructure resources, plans,
metrics, actions, and experiment results.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ResourceType(str, Enum):
    VM = "vm"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    STORAGE = "storage"


class ResourceStatus(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    TERMINATED = "terminated"
    MIGRATING = "migrating"


class ActionType(str, Enum):
    PROVISION = "provision"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE = "migrate"
    HEAL = "heal"
    TERMINATE = "terminate"
    RECONFIGURE = "reconfigure"


class ScaleDirection(str, Enum):
    UP = "scale_up"
    DOWN = "scale_down"
    MAINTAIN = "maintain"


class FailureType(str, Enum):
    INSTANCE_CRASH = "instance_crash"
    NETWORK_PARTITION = "network_partition"
    DISK_FULL = "disk_full"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    LATENCY_SPIKE = "latency_spike"
    DNS_FAILURE = "dns_failure"


class Region(str, Enum):
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    SIMULATED = "simulated"


class ScalingStrategy(str, Enum):
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    COST_AWARE = "cost_aware"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Resource specifications & state
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class ResourceSpec(BaseModel):
    """Specification for a cloud resource to be provisioned."""
    resource_type: ResourceType
    name: str = ""
    vcpus: int = 2
    memory_gb: float = 4.0
    storage_gb: float = 50.0
    region: Region = Region.US_EAST_1
    provider: CloudProvider = CloudProvider.SIMULATED
    replicas: int = 1
    tags: dict[str, str] = Field(default_factory=dict)


class Resource(BaseModel):
    """A live cloud resource with runtime state."""
    id: str = Field(default_factory=_new_id)
    spec: ResourceSpec
    status: ResourceStatus = ResourceStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cost_per_hour: float = 0.0
    metrics: ResourceMetrics | None = None

    def is_healthy(self) -> bool:
        return self.status in (ResourceStatus.RUNNING,)


class ResourceMetrics(BaseModel):
    """Point-in-time metrics for a resource."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Infrastructure state
# ---------------------------------------------------------------------------

class InfrastructureState(BaseModel):
    """Snapshot of all cloud infrastructure at a point in time."""
    resources: list[Resource] = Field(default_factory=list)
    total_cost_per_hour: float = 0.0
    total_vcpus: int = 0
    total_memory_gb: float = 0.0
    regions_active: list[Region] = Field(default_factory=list)
    tick: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def refresh_totals(self) -> None:
        running = [r for r in self.resources if r.status == ResourceStatus.RUNNING]
        self.total_cost_per_hour = sum(r.cost_per_hour for r in running)
        self.total_vcpus = sum(r.spec.vcpus * r.spec.replicas for r in running)
        self.total_memory_gb = sum(r.spec.memory_gb * r.spec.replicas for r in running)
        self.regions_active = list({r.spec.region for r in running})


# ---------------------------------------------------------------------------
# Actions & Plans
# ---------------------------------------------------------------------------

class InfraAction(BaseModel):
    """A single infrastructure action to be taken."""
    id: str = Field(default_factory=_new_id)
    action_type: ActionType
    target_resource_id: str | None = None
    resource_spec: ResourceSpec | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    estimated_cost_delta: float = 0.0
    priority: int = 0  # higher = more urgent


class InfrastructurePlan(BaseModel):
    """An ordered sequence of infrastructure actions to reach a goal state."""
    id: str = Field(default_factory=_new_id)
    actions: list[InfraAction] = Field(default_factory=list)
    goal_description: str = ""
    estimated_total_cost: float = 0.0
    estimated_duration_seconds: float = 0.0
    confidence: float = 0.0  # 0-1 how confident the planner is
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reasoning: str = ""


class ActionResult(BaseModel):
    """Outcome of executing a single InfraAction."""
    action_id: str
    success: bool
    message: str = ""
    resource_id: str | None = None
    actual_cost_delta: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Flight log (execution record)
# ---------------------------------------------------------------------------

class FlightLogEntry(BaseModel):
    """One step in a flight execution."""
    tick: int
    action: InfraAction
    result: ActionResult
    state_after: InfrastructureState | None = None


class FlightLog(BaseModel):
    """Complete record of a plan execution."""
    plan_id: str
    entries: list[FlightLogEntry] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    success: bool = False
    total_cost: float = 0.0
    replans: int = 0
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------

class Waypoint(BaseModel):
    """A desired infrastructure state to navigate toward."""
    id: str = Field(default_factory=_new_id)
    name: str
    compute_vcpus: int = 0
    memory_gb: float = 0.0
    replicas: int = 1
    regions: list[Region] = Field(default_factory=lambda: [Region.US_EAST_1])
    max_cost_per_hour: float = float("inf")
    min_availability: float = 0.99
    resource_types: list[ResourceType] = Field(default_factory=lambda: [ResourceType.VM])
    metadata: dict[str, Any] = Field(default_factory=dict)

    def distance_to(self, other: Waypoint) -> float:
        """Heuristic distance between two waypoints (higher = more different)."""
        cpu_diff = abs(self.compute_vcpus - other.compute_vcpus)
        mem_diff = abs(self.memory_gb - other.memory_gb)
        replica_diff = abs(self.replicas - other.replicas)
        region_overlap = len(set(self.regions) & set(other.regions))
        region_diff = max(len(self.regions), len(other.regions)) - region_overlap
        return float(cpu_diff + mem_diff + replica_diff * 5 + region_diff * 10)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class ScaleDecision(BaseModel):
    direction: ScaleDirection
    target_replicas: int | None = None
    target_vcpus: int | None = None
    target_memory_gb: float | None = None
    reason: str = ""
    confidence: float = 0.0


class ScaleResult(BaseModel):
    success: bool
    previous_replicas: int = 0
    new_replicas: int = 0
    cost_delta_per_hour: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

class MigrationPlan(BaseModel):
    id: str = Field(default_factory=_new_id)
    source_region: Region
    target_region: Region
    resource_ids: list[str] = Field(default_factory=list)
    estimated_downtime_seconds: float = 0.0
    estimated_data_transfer_gb: float = 0.0
    rollback_enabled: bool = True
    reason: str = ""


class MigrationResult(BaseModel):
    plan_id: str
    success: bool
    actual_downtime_seconds: float = 0.0
    data_transferred_gb: float = 0.0
    cost: float = 0.0
    message: str = ""
    rolled_back: bool = False


# ---------------------------------------------------------------------------
# Health & Healing
# ---------------------------------------------------------------------------

class HealthCheck(BaseModel):
    resource_id: str
    healthy: bool
    symptoms: list[str] = Field(default_factory=list)
    failure_type: FailureType | None = None
    severity: SeverityLevel = SeverityLevel.LOW
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Diagnosis(BaseModel):
    resource_id: str
    failure_type: FailureType
    severity: SeverityLevel
    root_cause: str = ""
    recommended_actions: list[ActionType] = Field(default_factory=list)
    confidence: float = 0.0


class HealResult(BaseModel):
    resource_id: str
    success: bool
    actions_taken: list[ActionType] = Field(default_factory=list)
    recovery_time_seconds: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Intelligence
# ---------------------------------------------------------------------------

class DemandForecast(BaseModel):
    """Predicted compute demand over a future horizon."""
    timestamps: list[int] = Field(default_factory=list)  # tick numbers
    predicted_cpu: list[float] = Field(default_factory=list)
    predicted_memory: list[float] = Field(default_factory=list)
    predicted_requests: list[float] = Field(default_factory=list)
    confidence_lower: list[float] = Field(default_factory=list)
    confidence_upper: list[float] = Field(default_factory=list)
    trend: str = "stable"  # rising, falling, stable, cyclic


class Anomaly(BaseModel):
    resource_id: str | None = None
    metric_name: str
    value: float
    expected_value: float
    z_score: float
    severity: SeverityLevel
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""


class OptimizedConfig(BaseModel):
    """Result of cost optimization."""
    original_cost_per_hour: float
    optimized_cost_per_hour: float
    savings_percent: float
    recommendations: list[InfraAction] = Field(default_factory=list)
    reasoning: str = ""


class CapacityPlan(BaseModel):
    """Long-term capacity plan."""
    horizon_ticks: int
    planned_resources: list[ResourceSpec] = Field(default_factory=list)
    estimated_cost_total: float = 0.0
    peak_vcpus: int = 0
    peak_memory_gb: float = 0.0
    headroom_percent: float = 20.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Configuration for running an experiment."""
    name: str
    duration_ticks: int = 100
    seed: int = 42
    scenario: str = "default"
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    """Result of a completed experiment."""
    config: ExperimentConfig
    success: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    flight_log: FlightLog | None = None
    summary: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimulationScenario(BaseModel):
    """Defines a cloud simulation scenario."""
    name: str
    description: str = ""
    total_ticks: int = 200
    base_load: float = 0.3
    load_pattern: str = "steady"  # steady, burst, cyclic, ramp, random
    failure_probability: float = 0.02
    failure_types: list[FailureType] = Field(
        default_factory=lambda: [FailureType.INSTANCE_CRASH, FailureType.MEMORY_LEAK]
    )
    region: Region = Region.US_EAST_1
    initial_resources: list[ResourceSpec] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
