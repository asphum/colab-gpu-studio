import type { GalleryAsset, GpuManifest } from '../lib/gpu-api';
import styles from '../gpu.module.css';

export function referenceAssets(assets: GalleryAsset[]): GalleryAsset[] {
  return assets.filter((asset) => !asset.mock && asset.metadata?.mock !== true && /^(image|video|audio)\//.test(asset.mime));
}

export function supportsReferences(manifest: GpuManifest | null): boolean {
  if (!manifest) return false;
  const declaration = manifest.assets as { required?: unknown[]; optional?: unknown[] } | undefined;
  if (declaration?.required?.length || declaration?.optional?.length) return true;
  const params = Array.isArray(manifest.parameters) ? manifest.parameters : Object.values(manifest.parameters || {});
  return params.some((p) => ['image', 'video', 'file', 'audio'].includes(String(p.type)) || p.widget === 'file');
}

export function ReferencePicker({ manifest, assets, selected, onChange, onUpload }: {
  manifest: GpuManifest | null; assets: GalleryAsset[]; selected: string[];
  onChange: (ids: string[]) => void; onUpload: (file: File) => Promise<void>;
}) {
  if (!manifest) return null;
  if (!supportsReferences(manifest)) return <p className="muted">เวิร์กโฟลว์นี้ไม่รองรับไฟล์อ้างอิง และจะไม่ส่งรูปภาพเข้าไปประมวลผล</p>;
  const available = referenceAssets(assets);
  return (
    <section className={styles.referencePicker} role="group" aria-label="ไฟล์อ้างอิง">
      <div className={styles.cardHead}>
        <div>
          <h2>ไฟล์อ้างอิง</h2>
          <p className={styles.cardSub}>เลือกรูปภาพหรือวิดีโออ้างอิงสำหรับเวิร์กโฟลว์นี้</p>
        </div>
        <label className="button secondary small">
          + อัปโหลดไฟล์
          <input
            hidden
            type="file"
            accept="image/*,video/*,audio/*"
            onChange={async (event) => {
              const file = event.target.files?.[0];
              event.target.value = '';
              if (file) await onUpload(file);
            }}
          />
        </label>
      </div>
      {available.length ? (
        <div className={styles.referenceList}>
          {available.map((asset) => (
            <label className={styles.referenceRow} key={asset.id}>
              <input
                type="checkbox"
                aria-label={`เลือก ${asset.filename || asset.id}`}
                checked={selected.includes(asset.id)}
                onChange={(event) =>
                  onChange(
                    event.target.checked
                      ? [...new Set([...selected, asset.id])]
                      : selected.filter((id) => id !== asset.id)
                  )
                }
              />
              {asset.mime.startsWith('image/') ? (
                <img src={asset.thumbnail_url || `/api/assets/${asset.id}/thumbnail`} alt="" loading="lazy" />
              ) : (
                <span className={styles.referenceMedia}>{asset.mime.startsWith('video/') ? 'วิดีโอ' : 'เสียง'}</span>
              )}
              <span className={styles.referenceName}>{asset.filename || asset.id}</span>
            </label>
          ))}
        </div>
      ) : (
        <div className="empty">
          <span className="emptyIcon">🖼</span>
          <strong>ยังไม่มีไฟล์อ้างอิง</strong>
          <span>คุณสามารถคลิกปุ่มอัปโหลดไฟล์ด้านบนเพื่อเพิ่มรูปภาพอ้างอิง</span>
        </div>
      )}
    </section>
  );
}
