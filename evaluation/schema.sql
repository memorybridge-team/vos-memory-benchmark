CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    configuration TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS native_runs (
    native_reference_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    dataset TEXT NOT NULL, video TEXT NOT NULL, object_id INTEGER NOT NULL,
    start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
    run_id INTEGER NOT NULL, seed INTEGER NOT NULL,
    evaluation_revision INTEGER NOT NULL, runtime_revision INTEGER NOT NULL,
    native_memory_revision INTEGER NOT NULL, memory_path TEXT NOT NULL,
    switch_signature TEXT NOT NULL, metadata TEXT NOT NULL,
    UNIQUE (experiment_id, dataset, video, object_id, start_frame, end_frame, run_id, seed, switch_signature)
);
CREATE TABLE IF NOT EXISTS native_switches (
    native_reference_id TEXT NOT NULL REFERENCES native_runs(native_reference_id) ON DELETE CASCADE,
    switch_name TEXT NOT NULL, switch_frame INTEGER NOT NULL,
    switch_seconds REAL, switch_gpu_mb REAL,
    PRIMARY KEY (native_reference_id, switch_name)
);
CREATE TABLE IF NOT EXISTS native_frame_scores (
    native_reference_id TEXT NOT NULL REFERENCES native_runs(native_reference_id) ON DELETE CASCADE,
    frame INTEGER NOT NULL, has_gt INTEGER NOT NULL CHECK (has_gt IN (0,1)),
    gt_visible INTEGER CHECK (gt_visible IN (0,1)), j REAL, f REAL, jf REAL,
    CHECK (has_gt = 1 OR (gt_visible IS NULL AND j IS NULL AND f IS NULL AND jf IS NULL)),
    PRIMARY KEY (native_reference_id, frame)
);
CREATE TABLE IF NOT EXISTS native_frame_times (
    native_reference_id TEXT NOT NULL REFERENCES native_runs(native_reference_id) ON DELETE CASCADE,
    frame INTEGER NOT NULL, seconds REAL, gpu_peak_mb REAL,
    PRIMARY KEY (native_reference_id, frame)
);
CREATE TABLE IF NOT EXISTS results (
    result_id INTEGER PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    native_reference_id TEXT NOT NULL REFERENCES native_runs(native_reference_id),
    dataset TEXT NOT NULL, video TEXT NOT NULL, object_id INTEGER NOT NULL,
    start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
    switch_name TEXT NOT NULL, switch_frame INTEGER NOT NULL,
    baseline TEXT NOT NULL, role TEXT NOT NULL, baseline_revision INTEGER NOT NULL,
    evaluation_revision INTEGER NOT NULL, runtime_revision INTEGER NOT NULL, cost_runtime_revision INTEGER,
    run_id INTEGER NOT NULL, seed INTEGER NOT NULL, j REAL, jf REAL, n_frames INTEGER NOT NULL,
    failure_rate REAL, switch_seconds REAL, switch_gpu_mb REAL,
    restoration_revision INTEGER, restoration_reference TEXT, restoration_measured_at INTEGER,
    restoration_status TEXT, extra_labels TEXT NOT NULL, metadata TEXT NOT NULL,
    UNIQUE (experiment_id, run_id, seed, dataset, video, object_id, start_frame, end_frame,
            switch_name, switch_frame, baseline, baseline_revision)
);
CREATE TABLE IF NOT EXISTS frame_scores (
    result_id INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    phase TEXT NOT NULL CHECK (phase IN ('pre','post')), frame INTEGER NOT NULL,
    frames_after_switch INTEGER NOT NULL, has_gt INTEGER NOT NULL CHECK (has_gt IN (0,1)),
    gt_visible INTEGER CHECK (gt_visible IN (0,1)), j REAL, f REAL, jf REAL,
    native_run_j REAL, native_run_f REAL, native_run_jf REAL,
    CHECK (has_gt = 1 OR (gt_visible IS NULL AND j IS NULL AND f IS NULL AND jf IS NULL)),
    PRIMARY KEY (result_id, phase, frame)
);
CREATE TABLE IF NOT EXISTS result_frame_times (
    result_id INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    frame INTEGER NOT NULL, seconds REAL, gpu_peak_mb REAL,
    PRIMARY KEY (result_id, frame)
);
CREATE TABLE IF NOT EXISTS restoration_frames (
    result_id INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    frame INTEGER NOT NULL, frames_before_switch INTEGER NOT NULL,
    native_is_cond INTEGER CHECK (native_is_cond IN (0,1)),
    target_is_cond INTEGER CHECK (target_is_cond IN (0,1)),
    PRIMARY KEY (result_id, frame)
);
CREATE TABLE IF NOT EXISTS restoration_fields (
    result_id INTEGER NOT NULL, frame INTEGER NOT NULL,
    field TEXT NOT NULL CHECK (field IN ('maskmem_features','obj_ptr')),
    r2 REAL, sse REAL, sst REAL, native_mean REAL, n_elements INTEGER NOT NULL,
    status TEXT NOT NULL, native_shape TEXT, target_shape TEXT,
    PRIMARY KEY (result_id, frame, field),
    FOREIGN KEY (result_id, frame) REFERENCES restoration_frames(result_id, frame) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    seed INTEGER NOT NULL, run_ids TEXT NOT NULL, native_statistic TEXT NOT NULL,
    recovery_statistic TEXT NOT NULL, restoration_revision INTEGER NOT NULL,
    configuration TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_frames (
    analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    result_id INTEGER NOT NULL, phase TEXT NOT NULL, frame INTEGER NOT NULL,
    native_statistic TEXT NOT NULL, native_j REAL, native_jf REAL, recovery_j REAL, recovery_jf REAL,
    native_reference_ready INTEGER NOT NULL CHECK (native_reference_ready IN (0,1)),
    native_reference_count INTEGER NOT NULL,
    PRIMARY KEY (analysis_id, result_id, phase, frame),
    FOREIGN KEY (result_id, phase, frame) REFERENCES frame_scores(result_id, phase, frame) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS recovery_summaries (
    analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    result_id INTEGER NOT NULL REFERENCES results(result_id) ON DELETE CASCADE,
    native_statistic TEXT NOT NULL,
    pre_recovery_j REAL, pre_recovery_jf REAL, post_recovery_j REAL, post_recovery_jf REAL,
    pre_recovery_j_n_frames INTEGER NOT NULL, pre_recovery_jf_n_frames INTEGER NOT NULL,
    post_recovery_j_n_frames INTEGER NOT NULL, post_recovery_jf_n_frames INTEGER NOT NULL,
    pre_zero_native_j_n_frames INTEGER NOT NULL, pre_zero_native_jf_n_frames INTEGER NOT NULL,
    post_zero_native_j_n_frames INTEGER NOT NULL, post_zero_native_jf_n_frames INTEGER NOT NULL,
    pre_missing_native_j_n_frames INTEGER NOT NULL, pre_missing_native_jf_n_frames INTEGER NOT NULL,
    post_missing_native_j_n_frames INTEGER NOT NULL, post_missing_native_jf_n_frames INTEGER NOT NULL,
    pre_pending_native_n_frames INTEGER NOT NULL, post_pending_native_n_frames INTEGER NOT NULL,
    statistics TEXT NOT NULL,
    PRIMARY KEY (analysis_id, result_id)
);
CREATE INDEX IF NOT EXISTS results_selection ON results(experiment_id, dataset, seed, run_id);
PRAGMA user_version = 1;
