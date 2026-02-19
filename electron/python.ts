import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';

const PYTHON_EXEC = process.platform === 'win32' ? 'python' : 'python3';

interface EffectPayload {
  projectId: string;
  strokeId: string;
  effectId: string;
  userInput: Record<string, unknown>;
  strokeInput: { path: number[][]; clicks: number[][] };
  canvasImageDataUrl: string;
}

/**
 * Runs a Python effect:
 * 1. Saves the canvas image as stroke input PNG
 * 2. Writes the instruction JSON with proper schema
 * 3. Spawns apply_effect.py
 * 4. Reads and returns the output image as a data URL
 */
export async function runPythonEffect(appRoot: string, payload: EffectPayload): Promise<string> {
  const { projectId, strokeId, effectId, userInput, strokeInput, canvasImageDataUrl } = payload;

  const projectDir = path.join(appRoot, 'project', projectId);
  const strokeDir = path.join(projectDir, 'stroke');
  await fs.mkdir(strokeDir, { recursive: true });

  // 1. Save canvas image as stroke input PNG
  const inputImagePath = path.join(strokeDir, `${strokeId}_input.png`);
  const base64Data = canvasImageDataUrl.replace(/^data:image\/\w+;base64,/, '');
  await fs.writeFile(inputImagePath, Buffer.from(base64Data, 'base64'));

  // 2. Write instruction JSON matching apply_effect.py expected schema
  const instructionPath = path.join(strokeDir, `${strokeId}_instructions.json`);
  const instructions = {
    effect_id: effectId,
    stroke_id: strokeId,
    project_id: projectId,
    user_input: userInput,
    stroke_input: {
      path: strokeInput.path,
      clicks: strokeInput.clicks,
    },
  };
  await fs.writeFile(instructionPath, JSON.stringify(instructions, null, 2));

  // 3. Save SVG representation of the stroke for archival/interop
  const svgPath = path.join(strokeDir, `${strokeId}.svg`);
  await fs.writeFile(svgPath, pathsToSvg(strokeInput.path, strokeInput.clicks));

  // 4. Spawn Python
  const scriptPath = path.join(appRoot, 'python', 'apply_effect.py');
  const TIMEOUT_MS = 3 * 60 * 1000; // 3 minutes
  console.log(`[PythonBridge] Running: ${PYTHON_EXEC} ${scriptPath} ${instructionPath}`);

  const exitCode = await new Promise<number>((resolve, reject) => {
    const proc = spawn(PYTHON_EXEC, [scriptPath, instructionPath], { cwd: appRoot });
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        console.error(`[PythonBridge] Process timed out after ${TIMEOUT_MS / 1000}s, killing...`);
        proc.kill('SIGTERM');
        setTimeout(() => proc.kill('SIGKILL'), 5000);
        resolve(1);
      }
    }, TIMEOUT_MS);

    proc.stdout.on('data', (d) => console.log(`[Python] ${d.toString().trim()}`));
    proc.stderr.on('data', (d) => console.error(`[Python ERR] ${d.toString().trim()}`));
    proc.on('close', (code) => { if (!settled) { settled = true; clearTimeout(timer); resolve(code ?? 1); } });
    proc.on('error', (err) => { if (!settled) { settled = true; clearTimeout(timer); reject(new Error(`Failed to spawn python: ${err.message}`)); } });
  });

  if (exitCode !== 0) {
    try {
      const result = JSON.parse(await fs.readFile(instructionPath, 'utf-8'));
      throw new Error(result.error || `Python exited with code ${exitCode}`);
    } catch (e) {
      if (e instanceof Error && e.message !== `Python exited with code ${exitCode}`) throw e;
      throw new Error(`Python exited with code ${exitCode}`);
    }
  }

  // 5. Verify success flag
  const resultJson = JSON.parse(await fs.readFile(instructionPath, 'utf-8'));
  if (!resultJson.effect_success) {
    throw new Error('Effect processing reported failure');
  }

  // 6. Read output image and return as data URL
  const outputPath = path.join(strokeDir, `${strokeId}_output.png`);
  const outputBuffer = await fs.readFile(outputPath);
  return `data:image/png;base64,${outputBuffer.toString('base64')}`;
}

/**
 * Scans python/ directory for effect *_requirements.json files.
 */
export async function loadEffectDefinitions(appRoot: string) {
  const pythonDir = path.join(appRoot, 'python');
  const entries = await fs.readdir(pythonDir, { withFileTypes: true });
  const effects = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const reqPath = path.join(pythonDir, entry.name, `${entry.name}_requirements.json`);
    try {
      const data = await fs.readFile(reqPath, 'utf-8');
      effects.push(JSON.parse(data));
    } catch {
      // skip dirs without requirements
    }
  }

  return effects;
}

function pathsToSvg(pathCoords: number[][], clicks: number[][]): string {
  let pathD = '';
  if (pathCoords.length > 0) {
    pathD = `M ${pathCoords[0][0]} ${pathCoords[0][1]}`;
    for (let i = 1; i < pathCoords.length; i++) {
      pathD += ` L ${pathCoords[i][0]} ${pathCoords[i][1]}`;
    }
  }
  const circles = (clicks || [])
    .map((c) => `  <circle cx="${c[0]}" cy="${c[1]}" r="3" fill="red"/>`)
    .join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path d="${pathD}" fill="none" stroke="black" stroke-width="2"/>
${circles}
</svg>`;
}
