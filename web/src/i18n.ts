/** English and Thai strings. Keys are shared; add both languages together. */

export type Lang = "en" | "th";

const en = {
  title: "Find your look in Blackbone HPL",
  lead: "Upload a photo of a room you like. We find each finished surface and show the Blackbone decors closest to its look.",
  choose: "Choose a photo",
  takePhoto: "Take a photo",
  example: "Try an example",
  drop: "or drop an image here",
  tips: "Works best with a well-lit photo that shows whole walls, islands or ceilings.",
  privacy: "Your photo is used only to find matches and is not stored.",
  previewNote: "Layout preview: results come from an example photo, whatever you upload.",
  stepModel: "Loading the matching model (first visit only)",
  stepSurfaces: "Finding surfaces",
  stepMatching: "Matching decors",
  materials: (n: number) => `${n} material${n === 1 ? "" : "s"} found`,
  material: (i: number) => `Material ${i}`,
  surfaces: (n: number) => `${n} surface${n === 1 ? "" : "s"}`,
  bestMatch: "Closest decor",
  others: "Other close looks",
  view: "View product",
  ask: "Ask about this decor",
  noEquivalent: (fam: string) => `Blackbone has no ${fam} look, so these are the closest decors.`,
  again: "Try another photo",
  errorTitle: "We couldn't read this photo",
  errorBody: "Try another photo, ideally a wider, well-lit shot of the room.",
  errorBusy: "Too many photos in a short time. Please wait a minute and try again.",
  errorBot: "Our bot check didn't pass. Reload the page and try again.",
  similarity: "match",
  family: {
    wood: "Wood", stone: "Stone", concrete: "Concrete", paint: "Painted", plaster: "Plaster",
    metal: "Metal", fabric: "Fabric", wallpaper: "Wallpaper", tile: "Tile", mosaic: "Mosaic",
    glass: "Glass", brick: "Brick", other: "Other",
  } as Record<string, string>,
};

const th: typeof en = {
  title: "ค้นหาลายที่ใช่จาก Blackbone HPL",
  lead: "อัปโหลดรูปห้องที่คุณชอบ เราจะหาพื้นผิวแต่ละส่วนในรูป แล้วแสดงลายลามิเนต Blackbone ที่ใกล้เคียงที่สุด",
  choose: "เลือกรูปภาพ",
  takePhoto: "ถ่ายรูป",
  example: "ลองใช้รูปตัวอย่าง",
  drop: "หรือลากรูปมาวางที่นี่",
  tips: "ได้ผลดีที่สุดกับรูปที่สว่างและเห็นผนัง เคาน์เตอร์ หรือฝ้าเต็มส่วน",
  privacy: "รูปของคุณใช้เพื่อค้นหาลายเท่านั้น และจะไม่ถูกเก็บไว้",
  previewNote: "ตัวอย่างหน้าจอ: ผลลัพธ์มาจากรูปตัวอย่าง ไม่ว่าคุณจะอัปโหลดรูปใด",
  stepModel: "กำลังโหลดโมเดลจับคู่ลาย (เฉพาะครั้งแรก)",
  stepSurfaces: "กำลังหาพื้นผิว",
  stepMatching: "กำลังจับคู่ลาย",
  materials: (n: number) => `พบวัสดุ ${n} แบบ`,
  material: (i: number) => `วัสดุที่ ${i}`,
  surfaces: (n: number) => `${n} พื้นผิว`,
  bestMatch: "ลายที่ใกล้เคียงที่สุด",
  others: "ลายใกล้เคียงอื่น ๆ",
  view: "ดูสินค้า",
  ask: "สอบถามลายนี้",
  noEquivalent: (fam: string) => `Blackbone ไม่มีลาย${fam} จึงแสดงลายที่ใกล้เคียงที่สุดแทน`,
  again: "ลองรูปอื่น",
  errorTitle: "อ่านรูปนี้ไม่ได้",
  errorBody: "ลองใช้รูปอื่น โดยเฉพาะรูปมุมกว้างที่มีแสงสว่างเพียงพอ",
  errorBusy: "ส่งรูปถี่เกินไป กรุณารอสักครู่แล้วลองใหม่",
  errorBot: "การตรวจสอบความปลอดภัยไม่ผ่าน กรุณาโหลดหน้าใหม่แล้วลองอีกครั้ง",
  similarity: "ความใกล้เคียง",
  family: {
    wood: "ไม้", stone: "หิน", concrete: "ปูนเปลือย", paint: "สีทาผนัง", plaster: "ปูนฉาบ",
    metal: "โลหะ", fabric: "ผ้า", wallpaper: "วอลเปเปอร์", tile: "กระเบื้อง", mosaic: "โมเสก",
    glass: "กระจก", brick: "อิฐ", other: "อื่น ๆ",
  },
};

export const STRINGS = { en, th } as const;
export type Strings = typeof en;

export function detectLang(): Lang {
  try {
    const saved = localStorage.getItem("lang");
    if (saved === "en" || saved === "th") return saved;
  } catch {
    /* storage blocked: fall through */
  }
  return navigator.language.toLowerCase().startsWith("th") ? "th" : "en";
}
