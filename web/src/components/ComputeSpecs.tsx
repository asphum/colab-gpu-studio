import type { ComputeSpec } from '../lib/gpu-api';

export function ComputeSpecs({ specs }: { specs: ComputeSpec[] }) {
  return (
    <div style={{ display: 'grid', gap: '8px', marginTop: '4px' }}>
      <div className="actions">
        {specs.map((spec, index) => {
          const key = spec.key;
          if (key && typeof key === 'object' && !Array.isArray(key)) {
            const value = key as Record<string, unknown>;
            const variant = String(value.variant || '');
            if (!['VARIANT_GPU', 'GPU'].includes(variant)) return null;
            const accelerator = String(value.accelerator || 'GPU ไม่ทราบชื่อ');
            const shape = ['SHAPE_HIGHMEM', 'HIGHMEM'].includes(String(value.shape)) ? 'หน่วยความจำสูง (High-RAM)' : 'หน่วยความจำมาตรฐาน';
            return (
              <span className="pill" key={JSON.stringify(key)}>
                {accelerator} · {shape} · {spec.eligible === true ? 'มีสิทธิ์ขอใช้งาน' : 'ยังไม่ยืนยันสิทธิ์'}
              </span>
            );
          }
          return (
            <span className="pill" key={spec.id || spec.name || index}>
              {spec.name || spec.gpu || 'GPU'} · {spec.available === true ? 'พร้อมใช้งาน' : spec.available === false ? 'ไม่พร้อมใช้งาน' : 'ไม่ทราบสถานะ'}{spec.reason ? ` · ${spec.reason}` : ''}
            </span>
          );
        })}
      </div>
      <p className="muted" style={{ fontSize: '12.5px', margin: 0 }}>
        สิทธิ์การขอใช้งานไม่ได้รับประกันโควตาแบบเรียลไทม์ การตรวจสอบนี้จะไม่สร้าง runtime และอุปกรณ์ที่ยังไม่ยืนยันสิทธิ์จะไม่ถูกเลือกโดยอัตโนมัติ
      </p>
    </div>
  );
}
