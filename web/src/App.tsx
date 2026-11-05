import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  apiMessage, assetUrl, cancelGpuJob, createGpuJob, createProject, deleteAsset, disconnectGoogle,
  getGpuComputeSpecs, getGpuGallery, getGpuJob, getGpuJobs, getGpuModels,
  getGpuPresets, getReferenceAssets, getGpuSettings, getGpuUsage, getGpuWorkflowSchema, getGpuWorkflows, getProjects, importGpuWorkflow,
  isLoggedIn, login, publishGpuWorkflow, retryGpuJob, startGoogleAuth, stopGpuCompute, updateAsset, uploadAsset,
  validateGpuWorkflow, rollbackGpuWorkflow, saveGpuPreset, type ComputeSpec, type GalleryAsset, type GpuJob, type GpuPreset,
  type GpuManifest, type GpuModel, type GpuParameter, type GpuSettings, type GpuUsage, type GpuWorkflow, type Project, type Provider,
} from './lib/gpu-api';
import { GpuParamForm } from './components/GpuParamForm';
import { ReferencePicker, referenceAssets, supportsReferences } from './components/ReferencePicker';
import { GoogleOAuthSetup } from './components/GoogleOAuthSetup';
import { ComputeSpecs } from './components/ComputeSpecs';
import styles from './gpu.module.css';

const tabs = ['สร้างสรรค์', 'งาน', 'ผลงาน', 'ตั้งค่า'] as const;
type Tab = typeof tabs[number];
const terminalStates = new Set(['completed', 'succeeded', 'failed', 'failed_final', 'cancelled', 'cancelled_local', 'blocked']);

function record(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function parameterList(manifest: GpuManifest | null): GpuParameter[] {
  if (!manifest?.parameters) return [];
  if (Array.isArray(manifest.parameters)) return manifest.parameters;
  return Object.entries(manifest.parameters).map(([key, item]) => {
    const normalized = item as GpuParameter;
    return { ...normalized, key, id: normalized.id || key };
  });
}

function parameterKey(item: GpuParameter, index: number) {
  return item.key || item.id || item.name || `parameter_${index + 1}`;
}

function defaultParams(manifest: GpuManifest | null) {
  return Object.fromEntries(
    parameterList(manifest).map((item, index) => [
      parameterKey(item, index),
      item.default !== undefined
        ? item.default
        : item.type === 'boolean'
        ? false
        : item.type === 'integer' || item.type === 'number'
        ? item.min ?? 0
        : '',
    ])
  );
}

function keyId() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `gpu-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function shown(value: unknown) {
  return value === null || value === undefined || value === '' ? 'ไม่ระบุ' : typeof value === 'object' ? JSON.stringify(value) : String(value);
}

function readableState(value?: string) {
  return value
    ? ({
        queued: 'อยู่ในคิว', pending: 'รอดำเนินการ', preparing: 'กำลังเตรียม', validating: 'กำลังตรวจสอบ',
        creating: 'กำลังสร้าง Runtime', reserved: 'สำรองแล้ว', active: 'Runtime กำลังทำงาน', started: 'เริ่มทำงานแล้ว',
        uploading: 'กำลังอัปโหลด', running: 'กำลังประมวลผล', saving: 'กำลังบันทึก', cleanup_pending: 'กำลังล้างข้อมูล',
        deleted: 'ปล่อยคืนแล้ว', quarantined: 'กักกันแล้ว', completed: 'เสร็จสมบูรณ์', succeeded: 'เสร็จสมบูรณ์',
        failed: 'ล้มเหลว', failed_final: 'ล้มเหลวถาวร', cancelled: 'ยกเลิกแล้ว', cancelled_local: 'ยกเลิกแล้ว', blocked: 'ถูกระงับ',
      }[value] || value)
    : 'ไม่ระบุ';
}

function workflowState(value?: string, active?: boolean) {
  return active ? 'เผยแพร่แล้ว · เวอร์ชันปัจจุบัน' : ({ draft: 'ฉบับร่าง', validated: 'ตรวจสอบแล้ว', published: 'เผยแพร่แล้ว · ประวัติ' }[value || ''] || value || 'ไม่ระบุ');
}

function progressPercent(value?: number) {
  return typeof value === 'number' ? Math.round(Math.max(0, Math.min(1, value)) * 100) : null;
}

function assetIdsFromParams(params: Record<string, unknown>, validIds: Set<string>) {
  return Object.values(params).filter((value): value is string => typeof value === 'string' && validIds.has(value));
}

function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      await login(password);
      onLoggedIn();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="loginCard" onSubmit={submit}>
        <h1>Colab GPU Studio</h1>
        <p className="muted">ใส่รหัสผ่านเพื่อเข้าสู่สตูดิโอส่วนตัว</p>
        <label>
          รหัสผ่านการเข้าถึง
          <input
            autoFocus
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            placeholder="กรุณากรอก STUDIO_PASSWORD"
            required
          />
        </label>
        {error && <p className="status error" role="alert">{error}</p>}
        <button className="button" disabled={busy} type="submit">
          {busy ? 'กำลังเข้าสู่ระบบ…' : 'เข้าสู่สตูดิโอ'}
        </button>
      </form>
    </main>
  );
}

function ProjectSheet({
  open,
  projects,
  projectId,
  onClose,
  onSelect,
  onCreate,
}: {
  open: boolean;
  projects: Project[];
  projectId: string;
  onClose: () => void;
  onSelect: (id: string) => void;
  onCreate: (title: string) => Promise<void>;
}) {
  const [title, setTitle] = useState('');
  const [creating, setCreating] = useState(false);

  if (!open) return null;

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    if (!title.trim()) return;
    setCreating(true);
    try {
      await onCreate(title.trim());
      setTitle('');
      onClose();
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className={styles.dialogBackdrop} role="presentation" onClick={onClose}>
      <section
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-label="จัดการโปรเจกต์"
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.cardHead}>
          <h2>จัดการโปรเจกต์</h2>
          <button className="button secondary" type="button" onClick={onClose}>เสร็จสิ้น</button>
        </div>
        <div className="field">
          <label>เลือกโปรเจกต์ปัจจุบัน</label>
          <select
            value={projectId}
            onChange={(e) => {
              onSelect(e.target.value);
              onClose();
            }}
          >
            <option value="">ยังไม่ได้เลือกโปรเจกต์</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>{p.title}</option>
            ))}
          </select>
        </div>
        <form className="field" onSubmit={handleCreate}>
          <label>สร้างโปรเจกต์ใหม่</label>
          <div className="actions" style={{ display: 'flex', gap: '2vw' }}>
            <input
              className="fieldInput"
              style={{ flex: 1 }}
              placeholder="ชื่อโปรเจกต์"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
            <button className="button" disabled={creating || !title.trim()} type="submit">
              {creating ? 'กำลังสร้าง…' : 'สร้าง'}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

function CreatePanel({
  projectId,
  models,
  workflows,
  presets,
  settings,
  manifest,
  params,
  setParams,
  gallery,
  selectedAssets,
  setSelectedAssets,
  uploadProgress,
  onUpload,
  modelId,
  setModelId,
  workflowId,
  setWorkflowId,
  runtime,
  setRuntime,
  maxRuntime,
  setMaxRuntime,
  loading,
  error,
  onSubmit,
  onApplyPreset,
  onSavePreset,
}: {
  projectId: string;
  models: GpuModel[];
  workflows: GpuWorkflow[];
  presets: GpuPreset[];
  settings: GpuSettings | null;
  manifest: GpuManifest | null;
  params: Record<string, unknown>;
  setParams: (value: Record<string, unknown>) => void;
  gallery: GalleryAsset[];
  selectedAssets: string[];
  setSelectedAssets: (value: string[]) => void;
  uploadProgress: Record<string, number>;
  onUpload: (file: File) => Promise<void>;
  modelId: string;
  setModelId: (value: string) => void;
  workflowId: string;
  setWorkflowId: (value: string) => void;
  runtime: Provider;
  setRuntime: (value: Provider) => void;
  maxRuntime: number;
  setMaxRuntime: (value: number) => void;
  loading: string;
  error: string;
  onSubmit: () => void;
  onApplyPreset: (preset: GpuPreset) => void;
  onSavePreset: (name: string) => void;
}) {
  const [presetName, setPresetName] = useState('');
  const [showPresetSave, setShowPresetSave] = useState(false);

  const workflowFamilies = useMemo(() => {
    const map = new Map<string, { modelId: string; items: GpuWorkflow[] }>();
    workflows.forEach((wf) => {
      if (wf.state !== 'published' && wf.active !== true) return;
      const list = map.get(wf.workflow_key)?.items || [];
      list.push(wf);
      list.sort((a, b) => String(b.workflow_version).localeCompare(String(a.workflow_version), undefined, { numeric: true }));
      map.set(wf.workflow_key, { modelId: wf.model_id, items: list });
    });
    return [...map.entries()];
  }, [workflows]);

  const disabledModels = useMemo(() => models.filter((m) => !m.enabled), [models]);
  const activeWorkflow = workflows.find((item) => item.id === workflowId);

  useEffect(() => {
    if (!workflowId && workflowFamilies.length > 0) {
      const firstFam = workflowFamilies[0][1];
      const defaultWf = firstFam.items.find((i) => i.active) || firstFam.items[0];
      if (defaultWf) {
        setModelId(defaultWf.model_id);
        setWorkflowId(defaultWf.id);
      }
    }
  }, [workflowFamilies, workflowId, setModelId, setWorkflowId]);

  if (!projectId) {
    return (
      <div className="empty">
        <strong>กรุณาเลือกหรือสร้างโปรเจกต์ที่แถบด้านบนก่อน</strong>
        <span>โปรเจกต์ใช้สำหรับจัดเก็บพรีเซ็ตพารามิเตอร์ ไฟล์นำเข้า และผลงานที่สร้าง</span>
      </div>
    );
  }

  const maxMinutesCap = typeof settings?.budget?.max_runtime_minutes === 'number'
    ? settings.budget.max_runtime_minutes
    : 60;

  return (
    <div className={styles.stack}>
      {/* 1. เลือกเวิร์กโฟลว์ */}
      <section className={styles.card}>
        <div className={styles.cardHead}>
          <h2>1. เลือกเวิร์กโฟลว์</h2>
          <span className="pill">{workflowFamilies.length} เวิร์กโฟลว์ที่ใช้ได้</span>
        </div>

        {workflowFamilies.length === 0 ? (
          <div className="empty">
            <strong>ยังไม่มีเวิร์กโฟลว์ที่เผยแพร่</strong>
            <span>กรุณาไปที่ “ตั้งค่า → จัดการเวิร์กโฟลว์” เพื่อเผยแพร่หรือกำหนดค่าเวิร์กโฟลว์ที่ต้องการ</span>
          </div>
        ) : (
          <div className={styles.stack} style={{ gap: '2.5vw' }}>
            {workflowFamilies.map(([key, fam]) => {
              const isSelected = activeWorkflow?.workflow_key === key;
              const currentWf = isSelected ? activeWorkflow : (fam.items.find((i) => i.active) || fam.items[0]);
              const model = models.find((m) => m.model_id === fam.modelId);

              return (
                <div
                  key={key}
                  className={styles.workflowCard}
                  data-selected={isSelected}
                  onClick={() => {
                    if (!isSelected && currentWf) {
                      setModelId(currentWf.model_id);
                      setWorkflowId(currentWf.id);
                    }
                  }}
                >
                  <div className={styles.workflowHeader}>
                    <div>
                      <strong>{key}</strong>
                      <p className="muted" style={{ margin: '0.5vw 0 0', fontSize: '3vw' }}>
                        โมเดล: {model?.display_name || fam.modelId}
                      </p>
                    </div>
                    {fam.items.length > 1 ? (
                      <select
                        aria-label="เปลี่ยนเวอร์ชัน"
                        value={isSelected ? workflowId : currentWf.id}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => {
                          const target = fam.items.find((i) => i.id === e.target.value);
                          if (target) {
                            setModelId(target.model_id);
                            setWorkflowId(target.id);
                          }
                        }}
                        style={{ width: 'auto', minHeight: '8vw', padding: '1vw 2vw' }}
                      >
                        {fam.items.map((i) => (
                          <option key={i.id} value={i.id}>
                            v{i.workflow_version} {i.active ? '（ปัจจุบัน）' : ''}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <span className="pill" style={{ fontSize: '2.6vw' }}>v{currentWf.workflow_version}</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {disabledModels.length > 0 && (
          <details className="foldable">
            <summary>โมเดลที่ยังไม่เปิดให้ใช้ ({disabledModels.length})</summary>
            <div className="foldContent">
              {disabledModels.map((m) => (
                <p key={m.model_id} className="muted" style={{ margin: 0, fontSize: '3vw' }}>
                  • <strong>{m.display_name}</strong>: {m.reason || 'รอผู้ดูแลระบบตรวจสอบก่อนเปิดใช้งาน'}
                </p>
              ))}
            </div>
          </details>
        )}
      </section>

      {/* 2. ตั้งค่าพารามิเตอร์ */}
      <section className={styles.card}>
        <div className={styles.cardHead}>
          <h2>2. ตั้งค่าพารามิเตอร์</h2>
          <details style={{ position: 'relative' }}>
            <summary className="pill" style={{ cursor: 'pointer' }}>จัดการพรีเซ็ต</summary>
            <div
              style={{
                position: 'absolute', right: 0, top: '100%', zIndex: 10, minWidth: '60vw',
                background: 'rgba(20, 28, 26, 0.95)', border: '1px solid var(--line-strong)',
                borderRadius: '3vw', padding: '3vw', marginTop: '1vw', backdropFilter: 'blur(16px)'
              }}
            >
              <div className="field">
                <label>ใช้พรีเซ็ต</label>
                <select
                  defaultValue=""
                  onChange={(e) => {
                    const preset = presets.find((p) => p.id === e.target.value);
                    if (preset) onApplyPreset(preset);
                  }}
                >
                  <option value="">เลือกพรีเซ็ต</option>
                  {presets.filter((p) => p.workflow_version_id === workflowId).map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </div>
              <div style={{ marginTop: '2vw' }}>
                {!showPresetSave ? (
                  <button className="button secondary" style={{ width: '100%', minHeight: '8vw' }} type="button" onClick={() => setShowPresetSave(true)}>
                    บันทึกเป็นพรีเซ็ตใหม่
                  </button>
                ) : (
                  <div style={{ display: 'flex', gap: '1.5vw' }}>
                    <input
                      className="fieldInput"
                      style={{ flex: 1, minHeight: '8vw', padding: '1vw 2vw' }}
                      placeholder="ชื่อพรีเซ็ต"
                      value={presetName}
                      onChange={(e) => setPresetName(e.target.value)}
                    />
                    <button
                      className="button"
                      style={{ minHeight: '8vw', padding: '1vw 3vw' }}
                      type="button"
                      disabled={!presetName.trim()}
                      onClick={() => {
                        onSavePreset(presetName.trim());
                        setPresetName('');
                        setShowPresetSave(false);
                      }}
                    >
                      บันทึก
                    </button>
                  </div>
                )}
              </div>
            </div>
          </details>
        </div>

        {manifest && (
          <GpuParamForm
            manifest={manifest}
            values={params}
            assets={gallery}
            uploadProgress={uploadProgress}
            onChange={(k, v) => setParams({ ...params, [k]: v })}
            onUpload={onUpload}
          />
        )}

        {supportsReferences(manifest) && (
          <ReferencePicker
            manifest={manifest}
            assets={gallery}
            selected={selectedAssets}
            onChange={setSelectedAssets}
            onUpload={onUpload}
          />
        )}
      </section>

      {/* 3. การสร้างและการประมวลผล */}
      <section className={styles.card}>
        <div className={styles.cardHead}>
          <h2>3. การสร้างและการประมวลผล</h2>
        </div>

        <div className="field">
          <label>สภาพแวดล้อมการประมวลผล</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2vw' }}>
            <button
              type="button"
              className={runtime === 'mock' ? 'button' : 'button secondary'}
              onClick={() => setRuntime('mock')}
            >
              จำลองออฟไลน์ (Mock)
            </button>
            <button
              type="button"
              className={runtime === 'colab' ? 'button' : 'button secondary'}
              disabled={!settings?.real_enabled}
              onClick={() => setRuntime('colab')}
            >
              Colab GPU {settings?.real_enabled ? '' : '（ยังไม่เปิดใช้งาน）'}
            </button>
          </div>
        </div>

        {runtime === 'colab' && (
          <div className="twoCol">
            <div className="field">
              <label>เวลาประมวลผลสูงสุด (นาที)</label>
              <input
                type="number"
                min={5}
                max={maxMinutesCap}
                value={maxRuntime}
                onChange={(e) => setMaxRuntime(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>สถานะโควตางบประมาณ</label>
              <div className="pill" style={{ width: '100%', minHeight: 'calc(var(--u) * 10)' }}>
                โควตารายวัน: {shown(settings?.budget?.daily_units)} CU
              </div>
            </div>
          </div>
        )}

        <button
          className="button"
          style={{ width: '100%', minHeight: 'calc(var(--u) * 12)', fontSize: 'clamp(4vw, 1vw + 1vh, 4.8vw)' }}
          disabled={!workflowId || loading === 'submit'}
          type="button"
          onClick={onSubmit}
        >
          {loading === 'submit' ? 'กำลังส่งงาน…' : 'เริ่มสร้างทันที'}
        </button>
        {error && <p className="status error" role="alert">{error}</p>}
      </section>
    </div>
  );
}

function TasksPanel({
  jobs,
  error,
  refreshing,
  onRefresh,
  onCancel,
  onRetry,
  onDetail,
}: {
  jobs: GpuJob[];
  error: string;
  refreshing: boolean;
  onRefresh: () => void;
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onDetail: (job: GpuJob) => void;
}) {
  return (
    <div className={styles.stack}>
      <section className={styles.card}>
        <div className={styles.sectionTitle}>
          <h2>งานในโปรเจกต์ปัจจุบัน</h2>
          <button className="button secondary" style={{ minHeight: '8vw', padding: '1vw 3vw' }} type="button" onClick={onRefresh}>
            {refreshing ? 'กำลังรีเฟรช…' : 'รีเฟรช'}
          </button>
        </div>
        {error && <p className="status error" role="alert">{error}</p>}
        {jobs.length ? (
          jobs.map((job) => (
            <article className={styles.job} key={job.id}>
              <div className={styles.jobHeader}>
                <strong>{job.workflow_key || job.id}</strong>
                <span className="pill">{job.provider === 'mock' ? 'MOCK · ' : ''}{readableState(job.state)}</span>
              </div>
              <div className={styles.jobMeta}>
                <span>งาน: {job.id.slice(0, 8)}</span>
                <span>สร้างเมื่อ: {job.created_at || 'เมื่อสักครู่'}</span>
                {job.estimated_units ? <span>ประเมิน: {shown(job.estimated_units)} CU</span> : null}
              </div>
              {progressPercent(job.progress) !== null && (
                <div className={styles.progress} aria-label={`ความคืบหน้า ${progressPercent(job.progress)}%`}>
                  <span style={{ width: `${progressPercent(job.progress)}%` }} />
                </div>
              )}
              {(job.error_message || job.error) && (
                <p className="status error" role="alert">{job.error_message || job.error}</p>
              )}
              <div className="actions">
                <button className="button secondary" type="button" onClick={() => onDetail(job)}>
                  ดูรายละเอียด
                </button>
                {!terminalStates.has(String(job.state || '').toLowerCase()) && (
                  <button className="button danger" type="button" onClick={() => onCancel(job.id)}>
                    ยกเลิก
                  </button>
                )}
                {String(job.state || '').toLowerCase().includes('failed') && (
                  <button className="button secondary" type="button" onClick={() => onRetry(job.id)}>
                    ลองใหม่
                  </button>
                )}
              </div>
            </article>
          ))
        ) : (
          <div className="empty">
            <strong>ยังไม่มีงานในโปรเจกต์นี้</strong>
            <span>เมื่องานสร้างถูกส่งจากแท็บสร้างสรรค์ รายการงานจะแสดงที่นี่</span>
          </div>
        )}
      </section>
    </div>
  );
}

function GalleryPanel({
  gallery,
  onFavorite,
  onDelete,
  onOpen,
}: {
  gallery: GalleryAsset[];
  onFavorite: (asset: GalleryAsset) => void;
  onDelete: (asset: GalleryAsset) => void;
  onOpen: (asset: GalleryAsset) => void;
}) {
  return (
    <section className={styles.card}>
      <div className={styles.cardHead}>
        <h2>แกลเลอรีผลงาน</h2>
        <span className="pill">{gallery.length} ผลงาน</span>
      </div>
      {gallery.length ? (
        <div className={styles.assetGrid}>
          {gallery.map((asset) => {
            const isVideo = asset.mime.startsWith('video/');
            const isJson = asset.mime.includes('json') || asset.filename?.toLowerCase().endsWith('.json');
            return (
              <article className={styles.asset} key={asset.id}>
                <button className={styles.assetImage} type="button" onClick={() => onOpen(asset)}>
                  {isVideo ? (
                    <video src={assetUrl(asset)} preload="metadata" muted aria-label={asset.filename || 'ไฟล์วิดีโอ'} />
                  ) : isJson ? (
                    <div className={styles.diagnosticsThumb}>JSON<br /><small>ไฟล์วินิจฉัย</small></div>
                  ) : (
                    <img src={assetUrl(asset, true)} alt={asset.filename || 'ผลงาน'} />
                  )}
                </button>
                <footer>
                  <span className={styles.assetName}>{asset.filename || asset.id}</span>
                  <div className="actions" style={{ gap: '1vw' }}>
                    <button className="button secondary" style={{ flex: 1, minHeight: '7vw', padding: '1vw' }} type="button" onClick={() => onFavorite(asset)}>
                      {asset.favorite ? 'เลิกถูกใจ' : 'ถูกใจ'}
                    </button>
                    <button className="button danger" style={{ minHeight: '7vw', padding: '1vw 2vw' }} type="button" onClick={() => onDelete(asset)}>
                      ลบ
                    </button>
                  </div>
                </footer>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty">
          <strong>ยังไม่มีผลงานที่สร้าง</strong>
          <span>ภาพและวิดีโอที่สร้างสำเร็จจะถูกบันทึกไว้ที่นี่โดยอัตโนมัติ</span>
        </div>
      )}
    </section>
  );
}

function WorkflowsPanel({
  workflows,
  onImport,
  onValidate,
  onPublish,
  onRollback,
}: {
  workflows: GpuWorkflow[];
  onImport: (bundle: string) => void;
  onValidate: (workflow: GpuWorkflow) => void;
  onPublish: (workflow: GpuWorkflow) => void;
  onRollback: (workflow: GpuWorkflow) => void;
}) {
  const [text, setText] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className={styles.stack}>
      <details className="foldable">
        <summary>นำเข้าร่างเวิร์กโฟลว์ใหม่</summary>
        <div className="foldContent">
          <textarea
            className={styles.code}
            placeholder="วาง JSON bundle ของเวิร์กโฟลว์ (รวม manifest, api_graph, object_info)"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="actions">
            <button className="button secondary" type="button" onClick={() => fileRef.current?.click()}>
              เลือกไฟล์ JSON
            </button>
            <input
              ref={fileRef}
              hidden
              type="file"
              accept="application/json,.json"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (file) setText(await file.text());
                e.target.value = '';
              }}
            />
            <button className="button" type="button" disabled={!text.trim()} onClick={() => onImport(text)}>
              นำเข้าร่าง
            </button>
          </div>
        </div>
      </details>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <h2>เวอร์ชันเวิร์กโฟลว์ที่ลงทะเบียนแล้ว</h2>
          <span className="pill">{workflows.length} เวอร์ชัน</span>
        </div>
        {workflows.length ? (
          workflows.map((wf) => (
            <article className={styles.workflowCard} key={wf.id}>
              <div className={styles.workflowHeader}>
                <div>
                  <strong>{wf.workflow_key} · v{wf.workflow_version}</strong>
                  <p className="muted" style={{ margin: '0.5vw 0 0', fontSize: '3vw' }}>
                    โมเดล {wf.model_id}
                  </p>
                </div>
                <span className="pill">{workflowState(wf.state, wf.active)}</span>
              </div>
              <div className="actions">
                <button
                  className="button secondary"
                  type="button"
                  disabled={wf.state === 'published'}
                  onClick={() => onValidate(wf)}
                >
                  ตรวจสอบ
                </button>
                <button
                  className="button"
                  type="button"
                  disabled={wf.state === 'published'}
                  onClick={() => onPublish(wf)}
                >
                  ยืนยันการเผยแพร่
                </button>
                <button
                  className="button secondary"
                  type="button"
                  disabled={wf.state !== 'published'}
                  onClick={() => onRollback(wf)}
                >
                  ย้อนกลับไปใช้เวอร์ชันนี้
                </button>
              </div>
            </article>
          ))
        ) : (
          <div className="empty">
            <strong>ตรวจไม่พบเวิร์กโฟลว์</strong>
          </div>
        )}
      </section>
    </div>
  );
}

function SettingsPanel({
  settings,
  usage,
  specs,
  workflows,
  loading,
  error,
  onConnect,
  onDisconnect,
  onCheck,
  onStop,
  onImport,
  onValidate,
  onPublish,
  onRollback,
  onLogout,
}: {
  settings: GpuSettings | null;
  usage: GpuUsage | null;
  specs: ComputeSpec[];
  workflows: GpuWorkflow[];
  loading: string;
  error: string;
  onConnect: (purpose: 'drive' | 'colab') => void;
  onDisconnect: (purpose: 'drive' | 'colab') => void;
  onCheck: () => void;
  onStop: () => void;
  onImport: (bundle: string) => void;
  onValidate: (workflow: GpuWorkflow) => void;
  onPublish: (workflow: GpuWorkflow) => void;
  onRollback: (workflow: GpuWorkflow) => void;
  onLogout: () => void;
}) {
  const [oauthConfigured, setOauthConfigured] = useState(false);

  return (
    <div className={styles.stack}>
      {/* 1. ภาพรวมการทำงานและโควตา */}
      <section className={styles.card}>
        <div className={styles.cardHead}>
          <h2>ภาพรวมการทำงานและโควตา</h2>
          <span className="pill">GPU จริง: {settings?.real_enabled ? 'เปิดใช้งานแล้ว' : 'ยังไม่เปิดใช้งาน'}</span>
        </div>
        <div className={styles.statGrid}>
          <div className={styles.stat}>
            <span>ประเมินวันนี้</span>
            <strong>{shown((usage as Record<string, unknown> | null)?.estimated_units)}</strong>
          </div>
          <div className={styles.stat}>
            <span>ใช้งานจริง</span>
            <strong>{shown((usage as Record<string, unknown> | null)?.actual_units)}</strong>
          </div>
          <div className={styles.stat}>
            <span>โควตารายวัน</span>
            <strong>{shown((usage as Record<string, unknown> | null)?.daily_limit)}</strong>
          </div>
          <div className={styles.stat}>
            <span>Runtime ที่ทำงานอยู่</span>
            <strong>{shown((usage as Record<string, unknown> | null)?.active_runtimes)}</strong>
          </div>
        </div>
      </section>

      {/* 2. การเชื่อมต่อ Google */}
      <section className={styles.card}>
        <div className={styles.cardHead}>
          <h2>การเชื่อมต่อ Google</h2>
        </div>
        {(['colab', 'drive'] as const).map((purpose) => {
          const state = settings?.google?.[purpose];
          return (
            <div className={styles.workflowCard} key={purpose}>
              <div className={styles.workflowHeader}>
                <strong>{purpose === 'colab' ? 'Google Colab' : 'Google Drive'}</strong>
                <span className="pill">
                  {state?.connected ? 'เชื่อมต่อแล้ว' : (oauthConfigured || state?.configured) ? 'ตั้งค่าแล้วยังไม่เชื่อมต่อ' : 'ยังไม่ได้ตั้งค่า'}
                </span>
              </div>
              <div className="actions">
                <button
                  className="button secondary"
                  type="button"
                  disabled={!oauthConfigured && !state?.configured}
                  onClick={() => onConnect(purpose)}
                >
                  เชื่อมต่อ {purpose === 'colab' ? 'Google Colab' : 'Google Drive'}
                </button>
                <button className="button danger" type="button" onClick={() => onDisconnect(purpose)}>
                  ยกเลิกการเชื่อมต่อ
                </button>
              </div>
            </div>
          );
        })}
        <details className="foldable">
          <summary>การตั้งค่าแอป Google OAuth</summary>
          <div className="foldContent">
            <GoogleOAuthSetup onConfigured={() => setOauthConfigured(true)} />
          </div>
        </details>
      </section>

      {/* 3. การจัดการทรัพยากร GPU */}
      <section className={styles.card}>
        <div className={styles.cardHead}>
          <h2>การจัดการทรัพยากร GPU</h2>
        </div>
        <div className="actions">
          <button className="button secondary" disabled={loading === 'check'} type="button" onClick={onCheck}>
            {loading === 'check' ? 'กำลังตรวจสอบ…' : 'ตรวจสอบ GPU ที่พร้อมใช้'}
          </button>
          <button className="button danger" disabled={loading === 'stop'} type="button" onClick={onStop}>
            {loading === 'stop' ? 'กำลังหยุดฉุกเฉิน…' : 'หยุดงานประมวลผลทั้งหมด'}
          </button>
        </div>
        {specs.length ? <ComputeSpecs specs={specs} /> : null}
      </section>

      {/* 4. จัดการเวิร์กโฟลว์และโมเดล */}
      <details className="foldable">
        <summary>จัดการเวิร์กโฟลว์และโมเดล</summary>
        <div className="foldContent">
          <WorkflowsPanel
            workflows={workflows}
            onImport={onImport}
            onValidate={onValidate}
            onPublish={onPublish}
            onRollback={onRollback}
          />
        </div>
      </details>

      {/* 5. ออกจากระบบ */}
      <div style={{ textAlign: 'center', marginTop: '2vw' }}>
        <button className="button danger" type="button" onClick={onLogout}>
          ออกจากระบบ
        </button>
      </div>

      {error && <p className="status error" role="alert">{error}</p>}
    </div>
  );
}

export default function App() {
  const [loggedIn, setLoggedIn] = useState(isLoggedIn());
  const [tab, setTab] = useState<Tab>('สร้างสรรค์');
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState('');
  const [showProjectSheet, setShowProjectSheet] = useState(false);
  const currentProjectId = useRef(projectId);
  currentProjectId.current = projectId;

  const [models, setModels] = useState<GpuModel[]>([]);
  const [workflows, setWorkflows] = useState<GpuWorkflow[]>([]);
  const [presets, setPresets] = useState<GpuPreset[]>([]);
  const [modelId, setModelId] = useState('');
  const [workflowId, setWorkflowId] = useState('');
  const [manifest, setManifest] = useState<GpuManifest | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [gallery, setGallery] = useState<GalleryAsset[]>([]);
  const [referenceImages, setReferenceImages] = useState<GalleryAsset[]>([]);
  const [jobs, setJobs] = useState<GpuJob[]>([]);
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const [runtime, setRuntime] = useState<Provider>('mock');
  const [maxRuntime, setMaxRuntime] = useState(30);
  const [settings, setSettings] = useState<GpuSettings | null>(null);
  const [usage, setUsage] = useState<GpuUsage | null>(null);
  const [specs, setSpecs] = useState<ComputeSpec[]>([]);
  const [error, setError] = useState('');
  const [jobError, setJobError] = useState('');
  const [loading, setLoading] = useState('');
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [confirm, setConfirm] = useState<'submit' | 'stop' | null>(null);
  const [detail, setDetail] = useState<GpuJob | null>(null);
  const [viewer, setViewer] = useState<GalleryAsset | null>(null);
  const [againJob, setAgainJob] = useState<GpuJob | null>(null);
  const [loginRefresh, setLoginRefresh] = useState(0);
  const [refreshingJobs, setRefreshingJobs] = useState(false);
  const idempotencyKey = useRef('');
  const retryKeys = useRef<Record<string, string>>({});
  const jobSequence = useRef(0);

  const activeProject = projects.find((p) => p.id === projectId);
  const activeWorkflow = workflows.find((item) => item.id === workflowId);

  const activeJobsCount = useMemo(() => {
    return jobs.filter((j) => !terminalStates.has(String(j.state || '').toLowerCase())).length;
  }, [jobs]);

  useEffect(() => {
    const expire = () => setLoggedIn(false);
    window.addEventListener('gpu-auth-expired', expire);
    return () => window.removeEventListener('gpu-auth-expired', expire);
  }, []);

  useEffect(() => {
    if (!loggedIn) return;
    let alive = true;
    Promise.allSettled([
      getProjects(),
      getGpuModels(),
      getGpuWorkflows(),
      getGpuPresets(),
      getGpuSettings(),
      getGpuUsage(),
    ]).then((results) => {
      if (!alive) return;
      const [projR, modR, wfR, preR, setR, useR] = results;
      if (projR.status === 'fulfilled') {
        setProjects(projR.value);
        setProjectId((curr) => curr || projR.value[0]?.id || '');
      } else setError(apiMessage(projR.reason));
      if (modR.status === 'fulfilled') setModels(modR.value);
      if (wfR.status === 'fulfilled') setWorkflows(wfR.value);
      if (preR.status === 'fulfilled') setPresets(preR.value);
      if (setR.status === 'fulfilled') setSettings(setR.value);
      if (useR.status === 'fulfilled') setUsage(useR.value);
    }).catch((cause) => alive && setError(apiMessage(cause)));
    return () => { alive = false; };
  }, [loggedIn, loginRefresh]);

  useEffect(() => {
    if (!modelId) return;
    const allowed = workflows.filter((w) => w.model_id === modelId);
    if (!allowed.some((w) => w.id === workflowId)) {
      setWorkflowId('');
      setManifest(null);
      setParams({});
    }
  }, [modelId, workflows, workflowId]);

  useEffect(() => {
    if (!workflowId) return;
    let alive = true;
    setLoading('schema');
    getGpuWorkflowSchema(workflowId).then((next) => {
      if (alive) {
        setManifest(next);
        setParams(defaultParams(next));
      }
    }).catch((cause) => alive && setError(`พารามิเตอร์ schema: ${apiMessage(cause)}`)).finally(() => alive && setLoading(''));
    return () => { alive = false; };
  }, [workflowId]);

  useEffect(() => {
    if (!projectId || !loggedIn) return;
    let alive = true;
    const refresh = async () => {
      const seq = ++jobSequence.current;
      setRefreshingJobs(true);
      try {
        const next = await getGpuJobs(projectId);
        if (!alive || seq !== jobSequence.current) return;
        setJobs(next.filter((j) => j.project_id === projectId));
        setJobError('');
      } catch (cause) {
        if (alive && seq === jobSequence.current) setJobError(apiMessage(cause));
      } finally {
        if (alive && seq === jobSequence.current) setRefreshingJobs(false);
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => {
      alive = false;
      window.clearInterval(timer);
      ++jobSequence.current;
    };
  }, [projectId, loggedIn]);

  useEffect(() => {
    setReferenceImages([]);
    setSelectedAssets([]);
    if (!projectId || !loggedIn) return;
    let alive = true;
    getReferenceAssets(projectId).then((next) => {
      if (alive) setReferenceImages(referenceAssets(next));
    }).catch((cause) => {
      if (alive) setError(`ไฟล์: ${apiMessage(cause)}`);
    });
    return () => { alive = false; };
  }, [projectId, loggedIn]);

  useEffect(() => {
    setSelectedAssets([]);
  }, [workflowId]);

  useEffect(() => {
    if (!projectId || !loggedIn) return;
    let alive = true;
    getGpuGallery(projectId).then((next) => alive && setGallery(next)).catch((cause) => alive && setError(`ผลงาน: ${apiMessage(cause)}`));
    return () => { alive = false; };
  }, [projectId, loggedIn, tab]);

  useEffect(() => {
    if (againJob && (againJob.resolved_params || againJob.params)) {
      setModelId(String(againJob.model_id || ''));
      const workflow = workflows.find((item) => item.workflow_key === againJob.workflow_key && String(item.workflow_version) === String(againJob.workflow_version));
      if (workflow) setWorkflowId(workflow.id);
      setParams({ ...(againJob.resolved_params || againJob.params) });
      setAgainJob(null);
      setTab('สร้างสรรค์');
    }
  }, [againJob, workflows]);

  async function createLocalProject(title: string) {
    try {
      const created = await createProject(title);
      setProjects((curr) => [created, ...curr]);
      setProjectId(created.id);
    } catch (cause) {
      setError(apiMessage(cause));
    }
  }

  async function refreshJobs() {
    if (!projectId) return;
    setRefreshingJobs(true);
    try {
      setJobs((await getGpuJobs(projectId)).filter((j) => j.project_id === projectId));
      setJobError('');
    } catch (cause) {
      setJobError(apiMessage(cause));
    } finally {
      setRefreshingJobs(false);
    }
  }

  async function handleUpload(file: File) {
    const uploadProjectId = projectId;
    if (!uploadProjectId) return;
    setUploadProgress((curr) => ({ ...curr, [file.name]: 0 }));
    try {
      const asset = await uploadAsset(uploadProjectId, file, (progress) => {
        if (currentProjectId.current === uploadProjectId) setUploadProgress((curr) => ({ ...curr, [file.name]: progress }));
      });
      if (currentProjectId.current !== uploadProjectId || asset.project_id !== uploadProjectId) return;
      setReferenceImages((curr) => [asset, ...curr.filter((item) => item.id !== asset.id)]);
      setSelectedAssets((curr) => [...new Set([...curr, asset.id])]);
    } catch (cause) {
      if (currentProjectId.current === uploadProjectId) setError(`อัปโหลด ${file.name}: ${apiMessage(cause)}`);
    } finally {
      if (currentProjectId.current === uploadProjectId) setUploadProgress((curr) => {
        const next = { ...curr };
        delete next[file.name];
        return next;
      });
    }
  }

  async function submitJob() {
    if (!projectId || !modelId || !workflowId || !activeWorkflow) return;
    if (!idempotencyKey.current) idempotencyKey.current = keyId();
    setLoading('submit');
    setError('');
    try {
      const validAssetIds = new Set(referenceAssets(referenceImages).filter((a) => a.project_id === projectId).map((a) => a.id));
      const body = {
        project_id: projectId,
        model_id: modelId,
        workflow_key: activeWorkflow.workflow_key,
        workflow_version: activeWorkflow.workflow_version,
        params,
        asset_ids: supportsReferences(manifest)
          ? [...new Set([...selectedAssets.filter((id) => validAssetIds.has(id)), ...assetIdsFromParams(params, validAssetIds)])]
          : [],
        execution_policy: {
          provider: runtime,
          max_runtime_minutes: maxRuntime,
          shutdown_after_job: true as const,
          idle_grace_minutes: 0 as const,
        },
      };
      await createGpuJob(body, idempotencyKey.current);
      setConfirm(null);
      setTab('งาน');
      idempotencyKey.current = '';
      await refreshJobs();
    } catch (cause) {
      setError(`ส่งงาน: ${apiMessage(cause)}`);
    } finally {
      setLoading('');
    }
  }

  async function action(name: string, operation: () => Promise<unknown>) {
    setLoading(name);
    setError('');
    try {
      await operation();
    } catch (cause) {
      setError(apiMessage(cause));
    } finally {
      setLoading('');
    }
  }

  async function importWorkflow(raw: string) {
    try {
      const value = JSON.parse(raw) as unknown;
      const source = record(value) && record(value.workflow) ? value.workflow : value;
      if (!record(source) || !record(source.manifest) || !record(source.api_graph) || !record(source.object_info)) {
        throw new Error('JSON ต้องประกอบด้วย manifest, api_graph, object_info');
      }
      await action('import', async () => {
        const workflow = await importGpuWorkflow(value as never);
        setWorkflows((curr) => [workflow, ...curr.filter((item) => item.id !== workflow.id)]);
        setError('นำเข้าเวิร์กโฟลว์เป็นฉบับร่างแล้ว กรุณาตรวจสอบก่อนเผยแพร่');
      });
    } catch (cause) {
      setError(`การนำเข้าล้มเหลว: ${apiMessage(cause)}`);
    }
  }

  async function reuseAsset(asset: GalleryAsset) {
    const sourceJobId = asset.source_job_id || asset.gpu_job_id;
    if (!sourceJobId) {
      setError('ไฟล์นี้ไม่มีบันทึกงานต้นฉบับ');
      return;
    }
    try {
      setAgainJob(await getGpuJob(sourceJobId));
    } catch (cause) {
      setError(`สร้างอีกครั้งล้มเหลว: ${apiMessage(cause)}`);
    }
  }

  async function favorite(asset: GalleryAsset) {
    try {
      const next = await updateAsset(asset.id, !asset.favorite);
      setGallery((curr) => curr.map((i) => i.id === next.id ? next : i));
    } catch (cause) {
      setError(apiMessage(cause));
    }
  }

  async function remove(asset: GalleryAsset) {
    try {
      await deleteAsset(asset.id);
      setGallery((curr) => curr.filter((i) => i.id !== asset.id));
      setViewer(null);
    } catch (cause) {
      setError(`ลบล้มเหลว: ${apiMessage(cause)}`);
    }
  }

  async function savePreset(name: string) {
    if (!workflowId) return;
    await action('preset', async () => {
      const next = await saveGpuPreset({ name, workflow_version_id: workflowId, overrides: params });
      setPresets((curr) => [next, ...curr]);
    });
  }

  function applyPreset(preset: GpuPreset) {
    if (preset.workflow_version_id !== workflowId) {
      setError('เวอร์ชันพรีเซ็ตไม่เข้ากัน');
      return;
    }
    setParams({ ...params, ...preset.overrides });
  }

  async function checkGpu() {
    await action('check', async () => setSpecs(await getGpuComputeSpecs(settings?.google?.colab?.connected ? 'colab' : 'mock')));
  }

  async function stopCompute() {
    await action('stop', async () => {
      await stopGpuCompute();
      setConfirm(null);
      setError('ส่งคำขอหยุดฉุกเฉินแล้ว');
    });
  }

  if (!loggedIn) {
    return <Login onLoggedIn={() => { setLoggedIn(true); setLoginRefresh((c) => c + 1); }} />;
  }

  return (
    <div className="app">
      {/* ส่วนหัว Topbar */}
      <header className="topbar">
        <div className="brand">
          Colab GPU Studio
          <small>v0.1.28</small>
        </div>
        <button
          className="projectChip"
          type="button"
          onClick={() => setShowProjectSheet(true)}
        >
          <span>📁 {activeProject ? activeProject.title : 'เลือกโปรเจกต์'}</span>
          <small style={{ color: 'var(--muted)', marginLeft: '1vw' }}>▾</small>
        </button>
      </header>

      {/* แถบนำทาง: รองรับทั้งมือถือและคอมพิวเตอร์ */}
      <nav className="bottomNav" aria-label="การนำทางในพื้นที่ทำงาน">
        {tabs.map((item) => {
          const isActive = tab === item;
          return (
            <button
              key={item}
              type="button"
              aria-current={isActive ? 'page' : undefined}
              onClick={() => setTab(item)}
            >
              <span>{item === 'สร้างสรรค์' ? '✦' : item === 'งาน' ? '⚡' : item === 'ผลงาน' ? '🖼' : '⚙'}</span>
              <span>{item}</span>
              {item === 'งาน' && activeJobsCount > 0 && (
                <span className="tabBadge">{activeJobsCount}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* หน้าต่างเลือกหรือสลับโปรเจกต์ */}
      <ProjectSheet
        open={showProjectSheet}
        projects={projects}
        projectId={projectId}
        onClose={() => setShowProjectSheet(false)}
        onSelect={setProjectId}
        onCreate={createLocalProject}
      />

      {/* พื้นที่ทำงานหลัก */}
      <main className={styles.page}>
        {tab === 'สร้างสรรค์' && (
          <CreatePanel
            projectId={projectId}
            models={models}
            workflows={workflows}
            presets={presets}
            settings={settings}
            manifest={manifest}
            params={params}
            setParams={setParams}
            gallery={referenceImages}
            selectedAssets={selectedAssets}
            setSelectedAssets={setSelectedAssets}
            uploadProgress={uploadProgress}
            onUpload={handleUpload}
            modelId={modelId}
            setModelId={setModelId}
            workflowId={workflowId}
            setWorkflowId={setWorkflowId}
            runtime={runtime}
            setRuntime={setRuntime}
            maxRuntime={maxRuntime}
            setMaxRuntime={setMaxRuntime}
            loading={loading}
            error={error}
            onSubmit={() => setConfirm('submit')}
            onApplyPreset={applyPreset}
            onSavePreset={(name) => void savePreset(name)}
          />
        )}

        {tab === 'งาน' && (
          <TasksPanel
            jobs={jobs}
            error={jobError}
            refreshing={refreshingJobs}
            onRefresh={() => void refreshJobs()}
            onCancel={(id) => void action('cancel', async () => { await cancelGpuJob(id); await refreshJobs(); })}
            onRetry={(id) => void action('retry', async () => {
              const job = jobs.find((item) => item.id === id);
              const key = retryKeys.current[id] || job?.idempotency_key || `retry:${id}`;
              retryKeys.current[id] = key;
              await retryGpuJob(id, key);
              await refreshJobs();
            })}
            onDetail={(job) => void action('inspect', async () => setDetail(await getGpuJob(job.id)))}
          />
        )}

        {tab === 'ผลงาน' && (
          <GalleryPanel
            gallery={gallery}
            onFavorite={(asset) => void favorite(asset)}
            onDelete={(asset) => void remove(asset)}
            onOpen={setViewer}
          />
        )}

        {tab === 'ตั้งค่า' && (
          <SettingsPanel
            settings={settings}
            usage={usage}
            specs={specs}
            workflows={workflows}
            loading={loading}
            error={error}
            onConnect={(purpose) => void action('connect', async () => {
              const url = await startGoogleAuth(purpose);
              if (url) window.location.href = url;
            })}
            onDisconnect={(purpose) => void action('disconnect', async () => {
              await disconnectGoogle(purpose);
              setSettings(await getGpuSettings());
            })}
            onCheck={() => void checkGpu()}
            onStop={() => setConfirm('stop')}
            onImport={(bundle) => void importWorkflow(bundle)}
            onValidate={(wf) => void action('validate', async () => {
              await validateGpuWorkflow(wf.id);
              setWorkflows(await getGpuWorkflows());
            })}
            onPublish={(wf) => void action('publish', async () => {
              await publishGpuWorkflow(wf.id);
              setWorkflows(await getGpuWorkflows());
            })}
            onRollback={(wf) => void action('rollback', async () => {
              await rollbackGpuWorkflow(wf.workflow_key, wf.workflow_version);
              setWorkflows(await getGpuWorkflows());
            })}
            onLogout={() => {
              sessionStorage.removeItem('colab-gpu-studio-token');
              setLoggedIn(false);
            }}
          />
        )}
      </main>

      {/* หน้าต่างยืนยันการเริ่มสร้าง */}
      {confirm === 'submit' && activeWorkflow && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setConfirm(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="ยืนยันการส่งงาน"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>ยืนยันการเริ่มสร้าง</h2>
            <p>
              เวิร์กโฟลว์: <strong>{activeWorkflow.workflow_key} · v{activeWorkflow.workflow_version}</strong><br />
              สภาพแวดล้อม: <strong>{runtime === 'colab' ? 'Colab GPU' : 'จำลองออฟไลน์'}</strong><br />
              {runtime === 'colab' && <>จำกัดเวลาสูงสุด: <strong>{maxRuntime} นาที</strong></>}
            </p>
            <div className="actions">
              <button className="button secondary" type="button" onClick={() => setConfirm(null)}>ยกเลิก</button>
              <button className="button" disabled={loading === 'submit'} type="button" onClick={() => void submitJob()}>
                {loading === 'submit' ? 'กำลังส่ง…' : 'ยืนยันเริ่มทำงาน'}
              </button>
            </div>
          </section>
        </div>
      )}

      {/* หน้าต่างยืนยันการหยุดฉุกเฉิน */}
      {confirm === 'stop' && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setConfirm(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="ยืนยันการหยุดฉุกเฉิน"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>ยืนยันการหยุดงานประมวลผลทั้งหมดฉุกเฉิน?</h2>
            <p className="error">การดำเนินการนี้จะขอยกเลิกงานทั้งหมดที่กำลังทำงานอยู่และปล่อยคืน Colab GPU runtime ที่เกี่ยวข้อง</p>
            <div className="actions">
              <button className="button secondary" type="button" onClick={() => setConfirm(null)}>ยกเลิก</button>
              <button className="button danger" type="button" onClick={() => void stopCompute()}>ยืนยันหยุดฉุกเฉิน</button>
            </div>
          </section>
        </div>
      )}

      {/* แผงรายละเอียดงาน */}
      {detail && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setDetail(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="รายละเอียดงาน"
            onClick={(e) => e.stopPropagation()}
          >
            <div className={styles.cardHead}>
              <h2>รายละเอียดงาน · {detail.id.slice(0, 8)}</h2>
              <button className="button secondary" type="button" onClick={() => setDetail(null)}>ปิด</button>
            </div>
            <p>
              สถานะ: <strong>{readableState(detail.state)}</strong> · สภาพแวดล้อม: <strong>{detail.provider}</strong><br />
              เวลาที่สร้าง: {String(detail.created_at || 'ไม่ทราบ')}
            </p>
            {detail.timings ? (
              <details className="foldable" open>
                <summary>บันทึกเวลาแต่ละขั้นตอน</summary>
                <div className="foldContent">
                  <pre className={styles.diagnostics}>{JSON.stringify(detail.timings, null, 2)}</pre>
                </div>
              </details>
            ) : null}
            {detail.resolved_params ? (
              <details className="foldable">
                <summary>สแนปชอตพารามิเตอร์</summary>
                <div className="foldContent">
                  <pre className={styles.diagnostics}>{JSON.stringify(detail.resolved_params, null, 2)}</pre>
                </div>
              </details>
            ) : null}
          </section>
        </div>
      )}

      {/* ตัวดูและจัดการผลงาน */}
      {viewer && (
        <div className={styles.dialogBackdrop} role="presentation" onClick={() => setViewer(null)}>
          <section
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-label="ดูผลงาน"
            onClick={(e) => e.stopPropagation()}
          >
            <div className={styles.cardHead}>
              <h2>{viewer.filename || viewer.id}</h2>
              <button className="button secondary" type="button" onClick={() => setViewer(null)}>ปิด</button>
            </div>
            {viewer.mime.startsWith('video/') ? (
              <video className={styles.viewerVideo} src={assetUrl(viewer)} controls autoPlay />
            ) : viewer.mime.startsWith('image/') ? (
              <img className={styles.viewerImage} src={assetUrl(viewer)} alt="" />
            ) : (
              <p className="muted">ไฟล์วินิจฉัยไม่สามารถดูตัวอย่างได้โดยตรง กรุณาดาวน์โหลดเพื่อเปิดดู</p>
            )}
            <div className="actions">
              <button className="button secondary" type="button" onClick={() => void favorite(viewer)}>
                {viewer.favorite ? 'เลิกถูกใจ' : 'ถูกใจ'}
              </button>
              <button className="button secondary" type="button" onClick={() => void reuseAsset(viewer)}>
                สร้างอีกครั้งด้วยพารามิเตอร์นี้
              </button>
              <a className="button secondary" href={assetUrl(viewer)} download={viewer.filename || viewer.id} target="_blank" rel="noreferrer">
                ดาวน์โหลดผลงาน
              </a>
              <button className="button danger" type="button" onClick={() => void remove(viewer)}>
                ลบผลงาน
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
