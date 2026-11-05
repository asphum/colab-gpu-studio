import { useMemo, useRef } from 'react';
import type { GpuManifest, GpuParameter, GalleryAsset } from '../lib/gpu-api';
import styles from './gpu-param-form.module.css';

type Props = {
  manifest?: GpuManifest | null;
  values: Record<string, unknown>;
  assets: GalleryAsset[];
  uploadProgress: Record<string, number>;
  onChange: (key: string, value: unknown) => void;
  onUpload: (file: File) => Promise<void>;
};

function parametersOf(manifest?: GpuManifest | null): GpuParameter[] {
  const raw = manifest?.parameters;
  if (Array.isArray(raw)) return raw;
  if (!raw || typeof raw !== 'object') return [];
  return Object.entries(raw).map(([key, parameter]) => { const normalized = parameter as GpuParameter; return { ...normalized, key, id: normalized.id || key }; });
}
function keyOf(parameter: GpuParameter, index: number) { return parameter.key || parameter.id || parameter.name || `parameter_${index + 1}`; }
function labelOf(parameter: GpuParameter, key: string) { return parameter.label || parameter.name || key; }
function optionOf(option: unknown): { value: string; label: string } {
  if (typeof option === 'object' && option !== null && 'value' in option) return { value: String(option.value), label: 'label' in option && option.label ? String(option.label) : String(option.value) };
  return { value: String(option), label: String(option) };
}
function initialValue(parameter: GpuParameter): unknown {
  if (parameter.default !== undefined) return parameter.default;
  if (parameter.widget === 'checkbox' || parameter.type === 'boolean') return false;
  if (parameter.type === 'integer' || parameter.type === 'number' || parameter.widget === 'number') return parameter.min ?? 0;
  return '';
}

export function GpuParamForm({ manifest, values, assets, uploadProgress, onChange, onUpload }: Props) {
  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});
  const parameters = useMemo(() => parametersOf(manifest), [manifest]);
  const regular = parameters.filter((parameter) => !parameter.advanced && parameter.group !== 'advanced');
  const advanced = parameters.filter((parameter) => parameter.advanced || parameter.group === 'advanced');
  const groups = useMemo(() => {
    const grouped = new Map<string, Array<{ parameter: GpuParameter; index: number }>>();
    regular.forEach((parameter, index) => { const group = parameter.group || 'พารามิเตอร์'; const list = grouped.get(group) || []; list.push({ parameter, index }); grouped.set(group, list); });
    return [...grouped.entries()];
  }, [regular]);

  function renderField(parameter: GpuParameter, index: number) {
    const key = keyOf(parameter, index);
    const label = labelOf(parameter, key);
    const current = values[key] === undefined ? initialValue(parameter) : values[key];
    const options = (parameter.options || []).map(optionOf);
    const widget = parameter.widget || (parameter.type === 'text' ? 'textarea' : parameter.type === 'boolean' ? 'checkbox' : parameter.type === 'integer' || parameter.type === 'number' ? 'number' : parameter.type === 'image' || parameter.type === 'video' || parameter.type === 'file' ? 'file' : options.length ? 'select' : 'text');
    return <div className={styles.field} key={key}>
      <label htmlFor={`gpu-${key}`}>{label}{parameter.required && <span aria-hidden="true"> · จำเป็น</span>}</label>
      {parameter.description && <span className={styles.hint}>{parameter.description}</span>}
      {widget === 'textarea' ? <textarea id={`gpu-${key}`} value={String(current ?? '')} onChange={(event) => onChange(key, event.target.value)} />
        : widget === 'select' ? <select id={`gpu-${key}`} value={String(current ?? '')} onChange={(event) => onChange(key, event.target.value)}><option value="">กรุณาเลือก</option>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select>
        : widget === 'checkbox' ? <label className={styles.checkbox}><input id={`gpu-${key}`} type="checkbox" checked={Boolean(current)} onChange={(event) => onChange(key, event.target.checked)} /><span>เปิดใช้งาน</span></label>
        : widget === 'number' ? <input id={`gpu-${key}`} type="number" value={String(current ?? '')} min={parameter.min} max={parameter.max} step={parameter.step || (parameter.type === 'integer' ? 1 : 'any')} onChange={(event) => onChange(key, parameter.type === 'integer' ? (event.target.value === '' ? '' : Number.parseInt(event.target.value, 10)) : (event.target.value === '' ? '' : Number(event.target.value)))} />
        : widget === 'file' ? <div className={styles.fileControl}><select id={`gpu-${key}`} value={String(current ?? '')} onChange={(event) => onChange(key, event.target.value)}><option value="">เลือกไฟล์ที่อัปโหลดแล้ว</option>{assets.filter((asset) => !parameter.type || parameter.type === 'file' || asset.mime.startsWith(`${parameter.type}/`)).map((asset) => <option key={asset.id} value={asset.id}>{asset.filename || asset.id}</option>)}</select><button type="button" className="button secondary" onClick={() => inputRefs.current[key]?.click()}>อัปโหลดไฟล์</button><input ref={(node) => { inputRefs.current[key] = node; }} type="file" hidden accept={parameter.type === 'image' ? 'image/*' : parameter.type === 'video' ? 'video/*' : undefined} onChange={async (event) => { const file = event.target.files?.[0]; if (file) { await onUpload(file); } event.target.value = ''; }} />{Object.entries(uploadProgress).map(([name, progress]) => <small className={styles.hint} key={name}>กำลังอัปโหลด {name} · {progress}%</small>)}</div>
        : <input id={`gpu-${key}`} type="text" value={String(current ?? '')} onChange={(event) => onChange(key, event.target.value)} />}
      {(parameter.min !== undefined || parameter.max !== undefined) && <span className={styles.hint}>ช่วงค่า: {parameter.min ?? 'ไม่จำกัด'} – {parameter.max ?? 'ไม่จำกัด'}</span>}
    </div>;
  }

  if (!manifest) return <div className="empty"><strong>ยังไม่ได้โหลดพารามิเตอร์ของเวิร์กโฟลว์</strong><span>หลังจากเลือกเวิร์กโฟลว์แล้ว ฟอร์มจะถูกสร้างขึ้นอัตโนมัติตาม manifest</span></div>;
  if (!parameters.length) return <div className="empty"><strong>เวิร์กโฟลว์นี้ไม่มีพารามิเตอร์ที่แก้ไขได้</strong><span>กรุณากลับไปที่แท็บเวิร์กโฟลว์เพื่อตรวจสอบ manifest</span></div>;
  return <div className={styles.form}>
    {groups.map(([group, fields]) => <section className={styles.group} key={group}><h3>{group}</h3>{fields.map(({ parameter, index }) => renderField(parameter, index))}</section>)}
    {advanced.length > 0 && <details className={styles.advanced}><summary>พารามิเตอร์ขั้นสูง (ซ่อนไว้เป็นค่าเริ่มต้น)</summary>{advanced.map((parameter, index) => renderField(parameter, index + regular.length))}</details>}
  </div>;
}
