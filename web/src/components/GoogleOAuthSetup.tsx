import { useEffect, useState } from 'react';
import { apiMessage, getGoogleClientConfig, saveGoogleClientConfig, type GoogleClientConfig } from '../lib/gpu-api';
import styles from '../gpu.module.css';

export function GoogleOAuthSetup({ onConfigured }: { onConfigured: () => void }) {
  const [config, setConfig] = useState<GoogleClientConfig | null>(null);
  const [clientId, setClientId] = useState('');
  const [clientSecret, setClientSecret] = useState('');
  const [webClient, setWebClient] = useState<Record<string, unknown> | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const callback = config?.redirect_uri || `${window.location.origin}/api/gpu/auth/google/callback`;
  useEffect(() => { let alive = true; getGoogleClientConfig().then((next) => { if (alive) setConfig(next); }).catch((cause) => { if (alive) setError(apiMessage(cause)); }); return () => { alive = false; }; }, []);
  async function save() {
    if (busy || !confirmed) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const next = await saveGoogleClientConfig(webClient ? { client_json: webClient, confirmed: true } : { client_id: clientId.trim(), client_secret: clientSecret, confirmed: true });
      setConfig(next); setClientId(''); setClientSecret(''); setWebClient(null); setConfirmed(false);
      setNotice('บันทึกการตั้งค่าแอปแบบเข้ารหัสเรียบร้อยแล้ว ขั้นตอนถัดไปให้คลิก "เชื่อมต่อ Google Colab" และ "เชื่อมต่อ Google Drive" เพื่อยืนยันสิทธิ์ของคุณเอง (ระบบจะไม่เปิด GPU อัตโนมัติ)');
      onConfigured();
    } catch (cause) { setError(apiMessage(cause)); } finally { setBusy(false); }
  }
  return <section className={styles.card} aria-label="การตั้งค่าแอป Google OAuth"><h2>การตั้งค่าแอป Google OAuth</h2>
    <p className="muted">{config?.configured ? 'แอปได้รับการตั้งค่าแล้ว การเชื่อมต่อบัญชีและการเปิดใช้งาน GPU เป็นขั้นตอนถัดไป' : 'ยังไม่ได้ตั้งค่า Google App จึงยังไม่สามารถข้ามไปยืนยันสิทธิ์ได้ คุณจำเป็นต้องสร้าง OAuth Web Client ของตนเองก่อน'}</p>
    <ol className={styles.oauthSteps}>
      <li>เปิด <a href="https://console.cloud.google.com/" target="_blank" rel="noreferrer">Google Cloud Console</a> แล้วสร้างหรือเลือกโปรเจกต์ของคุณ</li>
      <li>ในส่วน "APIs & Services → Library" ให้เปิดใช้งาน Google Drive API และเปิดใช้งาน Colab API ตามเอกสารทางการของ Colab</li>
      <li>ตั้งค่า Google Auth Platform: ใส่ชื่อแอป, เลือกกลุ่มเป้าหมายเป็น "ภายนอก (External)" หากอยู่ในโหมดทดสอบ ให้เพิ่มอีเมล Google ของคุณใน Test Users</li>
      <li>ในส่วน "Credentials" สร้าง OAuth client ID เลือกประเภท "Web application" ใส่ Callback URL ด้านล่าง แล้วดาวน์โหลดไฟล์ Client JSON</li>
      <li>นำเข้าไฟล์ JSON ที่นี่และบันทึก จากนั้นคลิกปุ่มเชื่อมต่อ Google ทั้งสองปุ่มด้านล่าง โดยใช้บัญชีที่มีสิทธิ์ใช้งาน Colab ในการอนุญาต</li>
    </ol>
    <label className="field">Callback URL สำหรับยืนยันสิทธิ์<input aria-label="Google OAuth Callback URL" readOnly value={callback} onFocus={(event) => event.target.select()} /></label>
    <button className="button secondary" type="button" onClick={async () => { try { await navigator.clipboard.writeText(callback); setNotice('คัดลอก Callback URL แล้ว'); } catch { setNotice('กรุณาคลิกที่กล่อง Callback URL แล้วเลือกทั้งหมดเพื่อคัดลอก'); } }}>คัดลอก Callback URL</button>
    <details><summary>สิทธิ์ OAuth (Scopes) ที่จำเป็น</summary><p className="muted">เพิ่ม scopes ต่อไปนี้ใน "Data Access / Scopes" ของ Google Auth Platform ระบบจะขอสิทธิ์เหล่านี้เมื่อเชื่อมต่อ ในโหมดทดสอบจะอนุญาตเฉพาะผู้ใช้ในรายชื่อ Test Users เท่านั้น</p><code className={styles.oauthScope}>https://www.googleapis.com/auth/colaboratory</code><code className={styles.oauthScope}>https://www.googleapis.com/auth/drive.file</code></details>
    <p className="muted">หากต้องระบุ "Authorized JavaScript origins" ให้ใช้ {window.location.origin} การยืนยันสิทธิ์จะไม่ทำให้เกิดค่าใช้จ่าย GPU และห้ามส่งไฟล์ JSON หรือ Client Secret ไปยังที่สาธารณะ</p>
    <details><summary>นำเข้า / เปลี่ยนแปลง OAuth Client</summary><div className={styles.stack}>
      <label className="button secondary">เลือกไฟล์ Google Client JSON<input data-testid="google-client-file" hidden type="file" accept="application/json,.json" onChange={async (event) => { const file = event.target.files?.[0]; event.target.value = ''; if (!file) return; try { if (file.size > 65536) throw new Error('ไฟล์ Client JSON มีขนาดใหญ่เกินไป'); const value: unknown = JSON.parse(await file.text()); if (!value || typeof value !== 'object' || !('web' in value)) throw new Error('กรุณาเลือกไฟล์ JSON ของ Web Application Client ไม่ใช่ Desktop Client หรือ Service Account Key'); setWebClient(value as Record<string, unknown>); setClientId(''); setClientSecret(''); setNotice('อ่านไฟล์แล้ว แต่ยังไม่ได้บันทึก'); setError(''); } catch (cause) { setWebClient(null); setError(apiMessage(cause)); } }} /></label>
      {!webClient && <><label className="field">หรือกรอก Client ID ด้วยตนเอง<input autoComplete="off" autoCapitalize="none" spellCheck={false} value={clientId} onChange={(event) => setClientId(event.target.value)} /></label><label className="field">Client Secret<input type="password" autoComplete="new-password" value={clientSecret} onChange={(event) => setClientSecret(event.target.value)} /></label></>}
      <label className={styles.oauthConfirm}><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>นี่คือ Google App ของฉันเอง ยืนยันให้บันทึกแบบเข้ารหัสไปยังเซิร์ฟเวอร์</span></label>
      <button className="button" type="button" disabled={busy || !confirmed || (!webClient && (!clientId || !clientSecret))} onClick={() => void save()}>{busy ? 'กำลังบันทึก…' : 'บันทึกการตั้งค่า OAuth App'}</button>
    </div></details>
    {error && <p className="status error" role="alert">{error}</p>}{notice && <p className="status" role="status">{notice}</p>}
  </section>;
}
