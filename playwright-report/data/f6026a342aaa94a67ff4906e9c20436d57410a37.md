# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: live-branch-real.spec.ts >> real live branch playback
- Location: tests\playwright\live-branch-real.spec.ts:3:5

# Error details

```
Error: expect(received).toMatchObject(expected)

- Expected  - 3
+ Received  + 3

  Object {
-   "playing": true,
-   "readyState": 4,
-   "srcObject": true,
+   "playing": false,
+   "readyState": 0,
+   "srcObject": false,
  }

Call Log:
- Timeout 60000ms exceeded while waiting on the predicate
```

# Page snapshot

```yaml
- generic [ref=e1]:
  - banner [ref=e2]:
    - generic [ref=e3]:
      - generic [ref=e4]: AI
      - generic [ref=e5]:
        - heading "دیوار عملیات هوش مصنوعی ویدیو" [level=1] [ref=e6]
        - generic [ref=e7]: نمایش برچسب‌های آتش، دود و پلاک پردازش‌شده در سرور
    - generic [ref=e8]:
      - generic [ref=e9]: پخش زنده است
      - link "مستندات API" [ref=e12] [cursor=pointer]:
        - /url: /docs
      - link "عیب‌یابی نرخ فریم" [ref=e13] [cursor=pointer]:
        - /url: /api/v1/diagnostics/fps?sample_seconds=5
      - button "نمایش تمام‌صفحه دیوار" [ref=e14] [cursor=pointer]
      - button "See live branch" [active] [pressed] [ref=e15] [cursor=pointer]
      - button "جریان هوش مصنوعی" [ref=e16] [cursor=pointer]
      - button "توقف پخش رابط کاربری" [ref=e17] [cursor=pointer]
  - generic [ref=e18]:
    - generic [ref=e19]:
      - heading "شاخه زنده GPU" [level=2] [ref=e20]
      - button "بازگشت به پیش‌نمایش" [ref=e21] [cursor=pointer]
    - alert [ref=e22]: مذاکره WHEP با کد 404 شکست خورد
    - generic [ref=e23]:
      - article [ref=e24] [cursor=pointer]:
        - heading "Camera 1 - Main Entrance" [level=3] [ref=e25]
        - generic [ref=e26]: "FPS: 18.0 · RES: 320×320 · wall"
        - generic [ref=e28]: در حال پخش زنده
      - article [ref=e29] [cursor=pointer]:
        - heading "Camera 2 - Reception" [level=3] [ref=e30]
        - generic [ref=e31]: "FPS: -- · RES: --"
        - generic [ref=e33]: "خطای بارگذاری منبع: خطای داخلی در سرویس رخ داد؛ دوباره تلاش کنید"
      - article [ref=e34] [cursor=pointer]:
        - heading "Camera 3 - Channel 101" [level=3] [ref=e35]
        - generic [ref=e36]: "FPS: 13.1 · RES: 320×320 · wall"
        - generic [ref=e38]: در حال پخش زنده
      - article [ref=e39] [cursor=pointer]:
        - heading "Camera 4 - Channel 201" [level=3] [ref=e40]
        - generic [ref=e41]: "FPS: 11.4 · RES: 320×320 · wall"
        - generic [ref=e43]: در حال پخش زنده
      - article [ref=e44] [cursor=pointer]:
        - heading "Camera 5 - Channel 301" [level=3] [ref=e45]
        - generic [ref=e46]: "FPS: -- · RES: --"
        - generic [ref=e48]: در حال دریافت…
      - article [ref=e49] [cursor=pointer]:
        - heading "Camera 6 - Channel 401" [level=3] [ref=e50]
        - generic [ref=e51]: "FPS: 11.4 · RES: 320×320 · wall"
        - generic [ref=e53]: در حال پخش زنده
      - article [ref=e54] [cursor=pointer]:
        - heading "Camera 7 - Channel 501" [level=3] [ref=e55]
        - generic [ref=e56]: "FPS: 15.7 · RES: 320×320 · wall"
        - generic [ref=e58]: در حال پخش زنده
      - article [ref=e59] [cursor=pointer]:
        - heading "Camera 8 - Channel 101" [level=3] [ref=e60]
        - generic [ref=e61]: "FPS: 19.8 · RES: 320×320 · wall"
        - generic [ref=e63]: در حال پخش زنده
      - article [ref=e64] [cursor=pointer]:
        - heading "Live branch sample 1" [level=3] [ref=e65]
        - generic [ref=e66]: "FPS: -- · RES: --"
        - generic [ref=e68]: "خطای بارگذاری منبع: مذاکره WHEP با کد 404 شکست خورد"
```

# Test source

```ts
  1  | import { test, expect } from "@playwright/test";
  2  | 
  3  | test("real live branch playback", async ({ page }) => {
  4  |   const errors: string[] = [];
  5  |   const failed: string[] = [];
  6  |   const protocol: string[] = [];
  7  |   page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  8  |   page.on("pageerror", error => errors.push(`pageerror: ${error.message}`));
  9  |   page.on("requestfailed", request => failed.push(`${request.method()} ${request.url()} ${request.failure()?.errorText}`));
  10 |   page.on("request", request => { if (/live-branch|whep/i.test(request.url())) protocol.push(`REQ ${request.method()} ${request.url()}`); });
  11 |   page.on("response", async response => {
  12 |     if (/live-branch|whep/i.test(response.url())) {
  13 |       protocol.push(`RES ${response.status()} ${response.url()}`);
  14 |       if (response.request().method() === "POST" && /whep/i.test(response.url())) protocol.push(`WHEP_BODY ${JSON.stringify((await response.text()).slice(0, 500))}`);
  15 |     }
  16 |   });
  17 |   await page.goto("/dashboard");
  18 |   await expect(page.locator("#liveBranchButton")).toBeVisible();
  19 |   const sources = await page.evaluate(async () => (await fetch("/api/v1/sources/preview-config")).json());
  20 |   const source = sources.sources.find((item: any) => item.source_uri === "file:///workspace/data/1.mp4" || item.source_uri === "/workspace/data/1.mp4");
  21 |   expect(source, `selected source identity not configured: ${JSON.stringify(sources)}`).toBeTruthy();
  22 |   await page.locator("#liveBranchButton").click();
  23 |   const card = page.locator('.live-branch-card').filter({ has: page.locator(`video[data-source-uri="${source.source_uri}"]`) }).first();
  24 |   await expect(card).toBeVisible({ timeout: 30000 });
  25 |   const video = card.locator("video");
> 26 |   await expect.poll(async () => await video.evaluate((v: HTMLVideoElement) => ({ readyState: v.readyState, width: v.videoWidth, height: v.videoHeight, playing: !v.paused, time: v.currentTime, srcObject: !!v.srcObject })), { timeout: 60000 }).toMatchObject({ readyState: 4, playing: true, srcObject: true });
     |                                                                                                                                                                                                                                                   ^ Error: expect(received).toMatchObject(expected)
  27 |   const before = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, rect: v.getBoundingClientRect().toJSON(), srcObject: !!v.srcObject }));
  28 |   await page.waitForTimeout(3000);
  29 |   const after = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, rect: v.getBoundingClientRect().toJSON(), playing: !v.paused, readyState: v.readyState, srcObject: !!v.srcObject }));
  30 |   expect(after.time).toBeGreaterThan(before.time);
  31 |   expect(after.rect.width).toBe(320); expect(after.rect.height).toBe(320);
  32 |   await card.click();
  33 |   await expect.poll(() => page.evaluate(() => !!document.fullscreenElement), { timeout: 30000 }).toBe(true);
  34 |   const fullBefore = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, playing: !v.paused, srcObject: !!v.srcObject }));
  35 |   await page.waitForTimeout(3000);
  36 |   const fullAfter = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, playing: !v.paused, srcObject: !!v.srcObject }));
  37 |   expect(fullAfter.time).toBeGreaterThan(fullBefore.time); expect(fullAfter.playing).toBe(true); expect(fullAfter.srcObject).toBe(true);
  38 |   await page.keyboard.press("Escape"); await page.locator("#liveBranchBack").click();
  39 |   await expect(page.locator("#liveBranchSection")).toBeHidden();
  40 |   console.log(JSON.stringify({ source: source.source_uri, before, after, fullBefore, fullAfter, protocol, errors, failed }, null, 2));
  41 |   expect(errors).toEqual([]); expect(failed).toEqual([]);
  42 | });
  43 | 
```