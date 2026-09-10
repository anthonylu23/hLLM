from __future__ import annotations

# ruff: noqa: F811 -- imported pytest fixture name
from datetime import timedelta
from pathlib import Path

import pytest
from hllm_control.models import (
    Backend,
    DType,
    MemoryBudget,
    MemoryDomain,
    ObjectiveWeights,
    PlannerSettings,
    PlanningMode,
    WorkerProfile,
)
from hllm_control.planner.measured import (
    BundleContent,
    DirectionEvidence,
    RequestSetupSample,
    WorkerEvidence,
    seal_bundle,
)
from hllm_control.planner.planner import assignments_for, create_plan
from hllm_control.profiling.link import LinkArtifact, LinkResult
from hllm_control.profiling.memory import PhysicalBudget
from hllm_control.profiling.models import (
    ComputeRunMeasurement,
    MemoryAmounts,
    ProfileKey,
    TimingRecord,
    digest,
    make_artifact,
)
from hllm_control.wire import deployment_plan_from_proto, deployment_plan_to_proto

from tests.process_helpers import write_model
from tests.python.test_link_profiling import artifact as link_fixture
from tests.python.test_profiling import conditions, key, memory  # noqa: F401 -- pytest fixture


def bundle_fixture(path: Path, key: ProfileKey):
    manifest = write_model(path / "four", family="llama")
    w = key.workload
    now = conditions().measured_at
    env = key.environment
    capacity = MemoryAmounts(host=10000000, device=1000)
    workers = tuple(
        WorkerEvidence(
            worker=WorkerProfile(
                worker_id=name,
                endpoint=name + ":1",
                backend=Backend.CPU,
                primary_memory_domain=MemoryDomain.HOST,
                supported_architectures=(manifest.architecture.architecture_id,),
                supported_execution_dtypes=(DType.F32,),
                memory_budgets=(MemoryBudget(domain=MemoryDomain.HOST, capacity_bytes=10000000),),
            ),
            memory_environment=env,
            compute_environment=env,
            runtime_binary_digest="a" * 64,
            runtime_device_identity="host",
            runtime_driver_version="not-applicable",
            admission_capacity=capacity,
            host=PhysicalBudget(
                available_bytes=10000000, headroom_bytes=100, extra_overhead_bytes=100
            ),
            observed_at=now,
            transport_mode="pageable",
        )
        for name in ("a", "b")
    )
    profiles = []
    for first, final in (("a", "b"), ("b", "a")):
        for split in range(1, manifest.config.num_layers):
            for a in assignments_for(first, final, split, manifest.config.num_layers):
                k = key.model_copy(
                    update={"assignment": a, "manifest_digest": manifest.manifest_digest}
                )
                profiles.append(
                    make_artifact(
                        k,
                        conditions(),
                        memory(k).model_copy(update={"admission_capacity": capacity}),
                    )
                )
                records = []
                # Asymmetric endpoints: b as driver has a large prefill penalty.
                for mode in ("whole-stage", "component-synchronized"):
                    for step in range(w.output_tokens):
                        base = (
                            (a.layer_end - a.layer_start) ** 2 + (50 if first == "b" else 0) + step
                        )
                        names = [("stage", None, base)]
                        if step == 0:
                            names.append(("sequence_allocation", None, 1))
                        if mode == "component-synchronized":
                            names += [
                                ("embedding" if a.owns_token_embedding else "from-wire", None, 0)
                            ]
                            names += [("layer", i, 0) for i in range(a.layer_start, a.layer_end)]
                            names += [
                                (c, None, 0)
                                for c in (
                                    ("final_norm", "lm_head", "sampling")
                                    if a.owns_sampling
                                    else ("to-wire",)
                                )
                            ]
                            names += [("runtime_overhead", None, base)]
                        records += [
                            TimingRecord(
                                cycle=0,
                                step=step,
                                context_tokens=0 if step == 0 else w.prompt_tokens + step - 1,
                                phase="prefill" if step == 0 else "decode",
                                component=c,  # pyright: ignore[reportArgumentType] -- validated fixture literals
                                layer_index=i,
                                timing_mode=mode,
                                milliseconds=ms,
                            )
                            for c, i, ms in names
                        ]
                profiles.append(
                    make_artifact(
                        k,
                        conditions().model_copy(
                            update={"process_policy": "loaded-stage-paired-passes"}
                        ),
                        ComputeRunMeasurement(
                            admission_capacity=capacity,
                            records=tuple(records),
                            identical_output_cycles=(0,),
                            completed=True,
                        ),
                    )
                )
    links = []
    directions = []
    for first, final in (("a", "b"), ("b", "a")):
        old = link_fixture()
        r = old.result
        assert r is not None
        source = r.source.model_copy(update={"worker_id": first})
        target = r.target.model_copy(update={"worker_id": final})
        result = LinkResult.model_validate(
            dict(
                source=source,
                target=target,
                channel_ready_ms=9999,
                prompt_tokens=w.prompt_tokens,
                hidden_size=manifest.config.hidden_size,
                output_tokens=w.output_tokens,
                warmup_cycles=0,
                measured_cycles=1,
                streams=[dict(cycle=0, setup_ms=9999, teardown_ms=9999)],
                samples=[
                    dict(
                        cycle=0,
                        step=s,
                        payload_bytes=(w.prompt_tokens if s == 0 else 1)
                        * manifest.config.hidden_size
                        * 2,
                        message_bytes=(w.prompt_tokens if s == 0 else 1)
                        * manifest.config.hidden_size
                        * 2
                        + 50,
                        feedback_bytes=20,
                        sender_encode_ms=1,
                        round_trip_ms=3 + s,
                    )
                    for s in range(w.output_tokens)
                ],
            )
        )
        data = old.model_dump(mode="json", exclude={"artifact_digest"})
        data.update(result=result.model_dump(mode="json"), concurrent_load="idle")
        link = LinkArtifact.model_validate({**data, "artifact_digest": digest(data)})
        links.append(link)
        directions.append(
            DirectionEvidence(
                source_worker_id=first,
                target_worker_id=final,
                source_probe=source,
                target_probe=target,
                path=link.paths[-1],
                link_digest=link.artifact_digest,
                request_setup_samples_ms=(2, 2, 2, 2, 2),
                request_setup_raw=tuple(
                    RequestSetupSample(client_ttft_ms=2, native_ttft_ms=1, native_setup_ms=1)
                    for _ in range(5)
                ),
                request_setup_evidence_digest=digest(
                    {
                        "samples": [
                            dict(client_ttft_ms=2.0, native_ttft_ms=1.0, native_setup_ms=1.0)
                            for _ in range(5)
                        ]
                    }
                ),
                request_setup_method="synthetic test only",
                request_setup_measured_at=now,
            )
        )
    return manifest, seal_bundle(
        BundleContent(
            manifest_digest=manifest.manifest_digest,
            checkpoint_digest=key.checkpoint_digest,
            workload=w,
            workers=workers,
            profiles=tuple(profiles),
            links=tuple(links),
            directions=tuple(directions),
            evaluated_at=now,
            concurrent_load="idle",
            input_kind=key.input_kind,
            input_digest=key.input_digest,
        )
    )


def report_for(manifest, bundle):
    return create_plan(
        manifest,
        [w.worker for w in bundle.workers],
        (),
        bundle.workload,
        PlannerSettings(
            mode=PlanningMode.MEASURED,
            execution_dtype=DType.F32,
            objective_weights=ObjectiveWeights(
                ttft=1, itl=bundle.workload.output_tokens - 1, pipeline_period=0, memory_pressure=0
            ),
        ),
        profile_bundle=bundle,
    )


def test_full_context_prediction_and_versioned_identity(tmp_path: Path, key: ProfileKey):
    manifest, bundle = bundle_fixture(tmp_path, key)
    report = report_for(manifest, bundle)
    assert report.plan is not None
    assert report.selected_candidate_id == "a--b-m002"
    p = next(
        c.performance for c in report.candidates if c.candidate_id == report.selected_candidate_id
    )
    assert p is not None
    assert p.ttft_ms == 16  # 2 setup + 2 allocation + 8 stage + 4 exchange
    assert p.decode_ms == (15, 18)
    assert p.generation_ms == 49
    assert deployment_plan_from_proto(deployment_plan_to_proto(report.plan)) == report.plan
    assert report_for(manifest, bundle).plan == report.plan
    assert (
        digest(report.plan.model_dump(mode="json", exclude={"plan_id", "plan_digest"}))
        == report.plan.plan_digest
    )
    with pytest.raises(ValueError, match="digest"):
        type(bundle).model_validate({**bundle.model_dump(), "checkpoint_digest": "e" * 64})


def test_unknown_physical_failure_and_scope(tmp_path: Path, key: ProfileKey):
    manifest, bundle = bundle_fixture(tmp_path, key)
    no_profiles = seal_bundle(
        BundleContent.model_validate(
            {**bundle.model_dump(exclude={"bundle_digest"}), "profiles": []}
        )
    )
    assert report_for(manifest, no_profiles).plan is None
    assert all(
        c.measurement_status == "unknown" for c in report_for(manifest, no_profiles).candidates
    )
    stale = seal_bundle(
        BundleContent.model_validate(
            {
                **bundle.model_dump(exclude={"bundle_digest"}),
                "evaluated_at": bundle.evaluated_at + timedelta(seconds=301),
            }
        )
    )
    assert report_for(manifest, stale).plan is None
    workers = tuple(
        b.model_copy(update={"host": b.host.model_copy(update={"available_bytes": 1})})
        for b in bundle.workers
    )
    unsafe = seal_bundle(
        BundleContent.model_validate(
            {**bundle.model_dump(exclude={"bundle_digest"}), "workers": workers}
        )
    )
    report = report_for(manifest, unsafe)
    assert report.plan is None
    assert all(c.measurement_status == "infeasible" for c in report.candidates)
    assert all(c.stages[0].pressure < 1 for c in report.candidates)


def test_changed_measurements_change_order_and_split(tmp_path: Path, key: ProfileKey):
    manifest, bundle = bundle_fixture(tmp_path, key)
    updated = []
    for p in bundle.profiles:
        m = p.measurement
        if isinstance(m, ComputeRunMeasurement) and p.key.assignment in assignments_for(
            "b", "a", 1, 4
        ):
            rows = tuple(
                r.model_copy(
                    update={
                        "milliseconds": 0.01 if r.component in ("stage", "runtime_overhead") else 0
                    }
                )
                for r in m.records
            )
            p = make_artifact(p.key, p.conditions, m.model_copy(update={"records": rows}))
        updated.append(p)
    new = seal_bundle(
        BundleContent.model_validate(
            {**bundle.model_dump(exclude={"bundle_digest"}), "profiles": updated}
        )
    )
    assert report_for(manifest, new).selected_candidate_id == "b--a-m001"
    assert report_for(manifest, new).plan != report_for(manifest, bundle).plan


def test_activation_refresh_rejects_memory_and_binary_changes(
    tmp_path: Path, key: ProfileKey, monkeypatch
):
    from hllm_control.planner import activation
    from hllm_control.proto import common_pb2, control_pb2, profile_pb2

    manifest, bundle = bundle_fixture(tmp_path, key)
    report = report_for(manifest, bundle)
    assert report.plan

    class Clock:
        @staticmethod
        def now(_zone):
            return bundle.evaluated_at

    monkeypatch.setattr(activation, "datetime", Clock)

    class Control:
        def __init__(self, binding):
            self.b = binding
            self.available = 10000000
            self.binary = binding.runtime_binary_digest

        def GetCapabilities(self, *args, **kwargs):
            return control_pb2.Capabilities(
                worker=profile_pb2.WorkerProfile(
                    worker_id=self.b.worker.worker_id,
                    endpoint=self.b.worker.endpoint,
                    backend=profile_pb2.BACKEND_CPU,
                    supported_architectures=[manifest.architecture.architecture_id],
                    supported_execution_dtypes=[common_pb2.DATA_TYPE_F32],
                    memory_budgets=[
                        profile_pb2.MemoryBudget(
                            domain=profile_pb2.MEMORY_DOMAIN_HOST, capacity_bytes=10000000
                        ),
                        profile_pb2.MemoryBudget(
                            domain=profile_pb2.MEMORY_DOMAIN_DEVICE, capacity_bytes=1000
                        ),
                    ],
                )
            )

        def GetQualificationState(self, *args, **kwargs):
            return control_pb2.QualificationState(
                available_host_bytes=self.available,
                boundary_transport_mode=self.b.transport_mode,
                binary_digest=self.binary,
                device_identity=self.b.runtime_device_identity,
                device_fingerprint=self.b.compute_environment.device_identity,
                driver_version=self.b.runtime_driver_version,
                backend_version=self.b.compute_environment.backend_version,
                allocator=self.b.compute_environment.allocator,
            )

        def GetMemoryReport(self, *args, **kwargs):
            return control_pb2.MemoryReport()

    controls = [Control(b) for b in bundle.workers]
    activation.validate_activation(manifest, report.plan, bundle, controls)  # pyright: ignore[reportArgumentType] -- structural RPC doubles
    controls[0].available = 1
    with pytest.raises(ValueError, match="fresh physical"):
        activation.validate_activation(manifest, report.plan, bundle, controls)  # pyright: ignore[reportArgumentType]
    controls[0].available = 10000000
    controls[0].binary = "c" * 64
    with pytest.raises(ValueError, match="binary/device"):
        activation.validate_activation(manifest, report.plan, bundle, controls)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("lost_reply", [False, True])
def test_failed_load_unwinds_only_acknowledged_or_uncertain_stages(
    tmp_path: Path, lost_reply: bool
):
    import grpc
    from hllm_control.controller import DeploymentSession
    from hllm_control.proto import common_pb2, control_pb2

    from tests.process_helpers import plan

    manifest = write_model(tmp_path / "load")
    deployment = plan(manifest)
    unloaded = []

    class Control:
        def __init__(self, index):
            self.index = index

        def LoadStage(self, *args, **kwargs):
            if self.index == 0:
                if lost_reply:
                    raise grpc.RpcError("lost response")
                return control_pb2.LoadStageResponse(accepted=False, detail="already loaded")
            return control_pb2.LoadStageResponse(accepted=True)

        def UnloadStage(self, *args, **kwargs):
            unloaded.append(self.index)
            return common_pb2.Empty()

    session = DeploymentSession(
        manifest, deployment, {s.worker_id: "127.0.0.1:1" for s in deployment.stages}
    )
    session._controls = [Control(0), Control(1)]  # pyright: ignore[reportAttributeAccessIssue] -- RPC test doubles
    with pytest.raises((RuntimeError, grpc.RpcError)):
        with session:
            raise AssertionError("failed setup must not enter session")
    assert unloaded == ([0, 1] if lost_reply else [1])


def test_disk_bundle_equivalence_and_integrity(tmp_path, key):
    from hllm_control.planner.measured import (
        DiskBundleContent,
        ProfileReference,
        read_profile_bundle,
        seal_disk_bundle,
    )

    manifest, bundle = bundle_fixture(tmp_path, key)
    folder = tmp_path / "profiles"
    folder.mkdir()
    refs = []
    for profile in bundle.profiles:
        (folder / (profile.artifact_digest + ".json")).write_text(profile.model_dump_json())
        refs.append(
            ProfileReference(
                artifact_digest=profile.artifact_digest, assignment=profile.key.assignment
            )
        )
    content = DiskBundleContent.model_validate(
        {
            **bundle.model_dump(mode="json", exclude={"profiles", "bundle_digest"}),
            "profile_references": refs,
        }
    )
    path = tmp_path / "bundle.json"
    disk = seal_disk_bundle(content, path)
    loaded = read_profile_bundle(path)
    before, after = report_for(manifest, bundle), report_for(manifest, loaded)
    assert before.candidates == after.candidates
    assert before.selected_candidate_id == after.selected_candidate_id
    assert after.plan is not None
    assert after.plan.profile_bundle_digest == disk.bundle_digest
    artifact = folder / (refs[0].artifact_digest + ".json")
    original = artifact.read_text()
    artifact.write_text("{}")
    with pytest.raises(ValueError):
        read_profile_bundle(path)
    # A bundle opened before tampering must also fail closed on subsequent access.
    with pytest.raises(ValueError):
        list(loaded.profiles_for(refs[0].assignment))
    artifact.write_text(original)
    wrong = content.model_copy(
        update={
            "profile_references": (
                refs[0].model_copy(update={"assignment": refs[-1].assignment}),
                *refs[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="reference mismatch"):
        seal_disk_bundle(wrong, tmp_path / "wrong.json")
    assert not (tmp_path / "wrong.json").exists()
    path.write_text(path.read_text().replace(disk.bundle_digest, "0" * 64))
    with pytest.raises(ValueError, match="bundle digest mismatch"):
        read_profile_bundle(path)
