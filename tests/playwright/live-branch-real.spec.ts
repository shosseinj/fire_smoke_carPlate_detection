import { test, expect } from "@playwright/test";

test("real live branch playback", async ({ page }) => {
  const errors: string[] = [];
  const failed: string[] = [];
  const protocol: string[] = [];
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  page.on("pageerror", error => errors.push(`pageerror: ${error.message}`));
  page.on("requestfailed", request => failed.push(`${request.method()} ${request.url()} ${request.failure()?.errorText}`));
  page.on("request", request => { if (/live-branch|whep/i.test(request.url())) protocol.push(`REQ ${request.method()} ${request.url()}`); });
  page.on("response", async response => {
    if (/live-branch|whep/i.test(response.url())) {
      protocol.push(`RES ${response.status()} ${response.url()}`);
      if (response.request().method() === "POST" && /whep/i.test(response.url())) protocol.push(`WHEP_BODY ${JSON.stringify((await response.text()).slice(0, 500))}`);
    }
  });
  await page.goto("/dashboard");
  await expect(page.locator("#liveBranchButton")).toBeVisible();
  const sources = await page.evaluate(async () => (await fetch("/api/v1/sources/preview-config")).json());
  const source = sources.sources.find((item: any) => item.source_uri === "file:///workspace/data/1.mp4" || item.source_uri === "/workspace/data/1.mp4");
  expect(source, `selected source identity not configured: ${JSON.stringify(sources)}`).toBeTruthy();
  await page.locator("#liveBranchButton").click();
  const card = page.locator('.live-branch-card').filter({ has: page.locator(`video[data-source-uri="${source.source_uri}"]`) }).first();
  await expect(card).toBeVisible({ timeout: 30000 });
  const video = card.locator("video");
  await expect.poll(async () => await video.evaluate((v: HTMLVideoElement) => ({ readyState: v.readyState, width: v.videoWidth, height: v.videoHeight, playing: !v.paused, time: v.currentTime, srcObject: !!v.srcObject })), { timeout: 60000 }).toMatchObject({ readyState: 4, playing: true, srcObject: true });
  const before = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, rect: v.getBoundingClientRect().toJSON(), srcObject: !!v.srcObject }));
  await page.waitForTimeout(3000);
  const after = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, rect: v.getBoundingClientRect().toJSON(), playing: !v.paused, readyState: v.readyState, srcObject: !!v.srcObject }));
  expect(after.time).toBeGreaterThan(before.time);
  expect(after.rect.width).toBe(320); expect(after.rect.height).toBe(320);
  await card.click();
  await expect.poll(() => page.evaluate(() => !!document.fullscreenElement), { timeout: 30000 }).toBe(true);
  const fullBefore = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, playing: !v.paused, srcObject: !!v.srcObject }));
  await page.waitForTimeout(3000);
  const fullAfter = await video.evaluate((v: HTMLVideoElement) => ({ time: v.currentTime, width: v.videoWidth, height: v.videoHeight, playing: !v.paused, srcObject: !!v.srcObject }));
  expect(fullAfter.time).toBeGreaterThan(fullBefore.time); expect(fullAfter.playing).toBe(true); expect(fullAfter.srcObject).toBe(true);
  await page.keyboard.press("Escape"); await page.locator("#liveBranchBack").click();
  await expect(page.locator("#liveBranchSection")).toBeHidden();
  console.log(JSON.stringify({ source: source.source_uri, before, after, fullBefore, fullAfter, protocol, errors, failed }, null, 2));
  expect(errors).toEqual([]); expect(failed).toEqual([]);
});
