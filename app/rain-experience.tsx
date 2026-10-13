"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GPUComputationRenderer } from "three/examples/jsm/misc/GPUComputationRenderer.js";

const initialRainfall = [
  2.1, 3.8, 9.9, 11.9, 0.0, 1.4, 0.7, 0.6, 4.1, 7.8, 6.6, 2.4,
  1.4, 1.6, 2.2, 3.8, 5.9, 7.4, 10.0, 8.8, 5.3, 3.2, 1.2, 0.4,
];

type Telemetry = {
  fps: number;
  drops: number;
  profile: string;
};

const rainVertexShader = /* glsl */ `
  attribute vec3 aOffset;
  attribute float aSpeed;
  attribute float aPhase;
  attribute float aLength;
  attribute float aGlow;
  attribute float aHour;
  attribute float aWidth;
  attribute float aVertexT;
  uniform float uTime;
  uniform float uRainfall[24];
  uniform vec2 uMouseWorld;
  uniform float uMouseForce;
  uniform vec2 uWind;
  uniform float uLightning;
  varying float vAlpha;
  varying float vGlow;
  varying float vTaper;
  varying float vLightning;

  void main() {
    float cycle = fract(uTime * aSpeed + aPhase);
    float height = 0.18 + (uRainfall[int(aHour)] / 12.0) * 8.4;
    float y = mix(height + 0.45, 0.03, cycle);
    vec3 p = vec3(aOffset.x, y + position.y * aLength, aOffset.z);
    vec2 delta = p.xz - uMouseWorld;
    float distanceSq = dot(delta, delta);
    float influence = exp(-distanceSq * 0.13) * uMouseForce;
    vec2 direction = normalize(delta + vec2(0.001));
    p.xz += direction * influence * (0.55 + cycle * 0.38);
    p.xz += uWind * (1.0 - cycle) * (0.15 + aLength * 0.16);
    // Tapered width: thick at leading edge (bottom), thin at trailing edge (top)
    float taperFactor = mix(1.0, 0.08, aVertexT);
    p.x += position.x * aWidth * taperFactor;
    float born = smoothstep(0.0, 0.06, cycle);
    float impact = 1.0 - smoothstep(0.90, 1.0, cycle);
    float depth = smoothstep(-2.6, 2.2, p.z);
    // Motion-blur alpha gradient: brighter at leading edge, fading toward tail
    float motionBlur = mix(0.35, 1.0, 1.0 - aVertexT);
    vAlpha = born * impact * mix(0.06, 0.38, aGlow) * mix(0.52, 1.0, depth) * motionBlur;
    vGlow = aGlow;
    vTaper = aVertexT;
    vLightning = uLightning;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const rainFragmentShader = /* glsl */ `
  varying float vAlpha;
  varying float vGlow;
  varying float vTaper;
  varying float vLightning;
  void main() {
    vec3 graphite = vec3(0.19, 0.22, 0.24);
    vec3 silver = vec3(0.52, 0.60, 0.64);
    vec3 bright = vec3(0.78, 0.84, 0.88);
    // Lightning illumination: flash makes drops much brighter
    vec3 baseColor = mix(graphite, silver, vGlow);
    vec3 litColor = mix(baseColor, bright, vLightning * 0.7);
    gl_FragColor = vec4(litColor, vAlpha * (1.0 + vLightning * 0.4));
  }
`;

const floorVertexShader = /* glsl */ `
  uniform sampler2D heightmap;
  uniform vec2 uTexel;
  varying vec3 vWorld;
  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vWave;

  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    float height = texture2D(heightmap, uv).x;
    float left = texture2D(heightmap, uv - vec2(uTexel.x, 0.0)).x;
    float right = texture2D(heightmap, uv + vec2(uTexel.x, 0.0)).x;
    float down = texture2D(heightmap, uv - vec2(0.0, uTexel.y)).x;
    float up = texture2D(heightmap, uv + vec2(0.0, uTexel.y)).x;
    world.y += height;
    vNormal = normalize(vec3((left - right) * 8.6, 1.0, (down - up) * 8.6));
    vWave = abs((left + right + down + up) - height * 4.0);
    vWorld = world.xyz;
    vUv = uv;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const floorFragmentShader = /* glsl */ `
  uniform float uLightning;
  varying vec3 vWorld;
  varying vec2 vUv;
  varying vec3 vNormal;
  varying float vWave;

  void main() {
    float waterField = 1.0;
    vec2 p = vWorld.xz;
    vec3 normal = normalize(vNormal);
    vec3 viewDirection = normalize(cameraPosition - vWorld);
    vec3 lightDirection = normalize(vec3(-0.42, 1.0, 0.38));
    float fresnel = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.2);
    vec3 reflected = reflect(-viewDirection, normal);
    float specular = pow(max(dot(reflect(-lightDirection, normal), viewDirection), 0.0), 78.0);
    float slope = clamp(length(normal.xz) * 3.7, 0.0, 1.0);
    float waveContour = smoothstep(0.00025, 0.0035, vWave);
    float readableCrest = smoothstep(0.018, 0.09, slope);
    float edgeFade = smoothstep(0.0, 0.08, vUv.x)
      * smoothstep(0.0, 0.08, 1.0 - vUv.x)
      * smoothstep(0.0, 0.12, vUv.y)
      * smoothstep(0.0, 0.12, 1.0 - vUv.y);
    vec3 base = vec3(0.0006, 0.0011, 0.0015);
    vec3 color = base;
    vec3 reflectedSky = mix(vec3(0.001, 0.003, 0.004), vec3(0.042, 0.069, 0.079),
      smoothstep(-0.15, 0.8, reflected.y));
    color += vec3(0.018, 0.026, 0.031) * fresnel * waterField;
    color += reflectedSky * fresnel * 0.68 * waterField;
    color += vec3(0.25, 0.31, 0.34) * specular * waterField;
    color += vec3(0.032, 0.049, 0.056) * slope * waterField;
    color += vec3(0.034, 0.052, 0.060) * waveContour * (0.65 + slope) * waterField;
    color += vec3(0.018, 0.029, 0.034) * readableCrest * waterField;
    // Lightning illumination on water surface
    color += vec3(0.06, 0.08, 0.09) * uLightning * fresnel;
    color += vec3(0.12, 0.14, 0.16) * uLightning * specular * 2.0;
    color *= edgeFade;
    gl_FragColor = vec4(color, 1.0);
  }
`;

const waterComputeShader = /* glsl */ `
  uniform float uTime;
  uniform vec4 uImpacts[16];

  void main() {
    vec2 cell = 1.0 / resolution.xy;
    vec2 uv = gl_FragCoord.xy * cell;
    vec4 value = texture2D(heightmap, uv);
    float north = texture2D(heightmap, uv + vec2(0.0, cell.y)).x;
    float south = texture2D(heightmap, uv - vec2(0.0, cell.y)).x;
    float east = texture2D(heightmap, uv + vec2(cell.x, 0.0)).x;
    float west = texture2D(heightmap, uv - vec2(cell.x, 0.0)).x;

    // Evan Wallace's water state layout: x = height, y = vertical velocity.
    // The velocity relaxes towards the four-neighbour average, then advances the
    // height. This preserves a clean circular wavefront instead of noisy pulses.
    float average = (north + south + east + west) * 0.25;
    float nextVelocity = (value.y + (average - value.x) * 0.16) * 0.9992;
    float nextHeight = value.x + nextVelocity;

    vec2 world = (uv - 0.5) * vec2(48.0, 22.0);
    float splat = 0.0;
    for (int i = 0; i < 16; i++) {
      float age = uTime - uImpacts[i].z;
      float live = step(0.0, age) * (1.0 - step(0.014, age));
      vec2 delta = world - uImpacts[i].xy;
      float radial = max(0.0, 1.0 - length(delta) / 0.62);
      radial = 0.5 - cos(radial * 3.14159265) * 0.5;
      splat += radial * uImpacts[i].w * live;
    }
    nextHeight += splat;

    value.y = nextVelocity;
    value.x = clamp(nextHeight, -0.14, 0.14);
    gl_FragColor = value;
  }
`;

const impactVertexShader = /* glsl */ `
  attribute vec3 aOffset;
  attribute float aPhase;
  attribute float aSize;
  uniform float uTime;
  uniform float uPixelRatio;
  varying float vPulse;

  void main() {
    float life = fract(uTime * 0.31 + aPhase);
    float pulse = smoothstep(0.89, 0.945, life) * (1.0 - smoothstep(0.945, 1.0, life));
    vec4 mv = modelViewMatrix * vec4(aOffset.x, 0.024, aOffset.z, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(8.0, aSize * pulse * uPixelRatio * (95.0 / -mv.z));
    vPulse = pulse;
  }
`;

const impactFragmentShader = /* glsl */ `
  varying float vPulse;
  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float d = length(p);
    float core = smoothstep(0.44, 0.0, d);
    float rim = smoothstep(0.035, 0.0, abs(d - 0.31));
    gl_FragColor = vec4(vec3(0.30, 0.39, 0.43), (core * 0.12 + rim * 0.28) * vPulse);
  }
`;

// --- Splash particles: burst outward when rain drops hit the ground ---
const splashVertexShader = /* glsl */ `
  attribute vec3 aOrigin;
  attribute vec3 aVelocity;
  attribute float aBirthTime;
  attribute float aLifespan;
  attribute float aSize;
  uniform float uTime;
  uniform float uPixelRatio;
  uniform float uLightning;
  varying float vLife;
  varying float vLightning;

  void main() {
    float age = uTime - aBirthTime;
    float t = clamp(age / aLifespan, 0.0, 1.0);
    // Dead particles: push off-screen
    if (t >= 1.0 || age < 0.0) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
      vLife = 0.0;
      vLightning = 0.0;
      return;
    }
    // Parabolic arc: rise then fall with gravity
    vec3 pos = aOrigin + aVelocity * age + vec3(0.0, -4.9 * age * age, 0.0);
    pos.y = max(0.01, pos.y);
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    float fade = 1.0 - t * t;
    gl_PointSize = max(0.5, aSize * fade * uPixelRatio * (60.0 / -mv.z));
    vLife = fade;
    vLightning = uLightning;
  }
`;

const splashFragmentShader = /* glsl */ `
  varying float vLife;
  varying float vLightning;
  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float d = length(p);
    float circle = smoothstep(0.5, 0.15, d);
    vec3 color = mix(vec3(0.35, 0.42, 0.48), vec3(0.62, 0.70, 0.76), vLife);
    color += vec3(0.15, 0.18, 0.20) * vLightning;
    gl_FragColor = vec4(color, circle * vLife * 0.42);
  }
`;

// --- Fog layer shader: subtle atmospheric depth ---
const fogLayerVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fogLayerFragmentShader = /* glsl */ `
  uniform float uTime;
  uniform float uLightning;
  varying vec2 vUv;
  void main() {
    float drift = sin(vUv.x * 3.0 + uTime * 0.08) * 0.5 + 0.5;
    float density = smoothstep(0.0, 0.5, vUv.y) * smoothstep(1.0, 0.55, vUv.y);
    density *= 0.06 + drift * 0.03;
    // Edge fade
    density *= smoothstep(0.0, 0.15, vUv.x) * smoothstep(1.0, 0.85, vUv.x);
    vec3 fogColor = vec3(0.12, 0.14, 0.16) + vec3(0.08, 0.10, 0.12) * uLightning;
    gl_FragColor = vec4(fogColor, density);
  }
`;

function weightedHour(data: number[], total: number) {
  let marker = Math.random() * total;
  for (let i = 0; i < data.length; i++) {
    marker -= Math.max(0.08, data[i]);
    if (marker <= 0) return i;
  }
  return data.length - 1;
}

function makeRainGeometry(maxDrops: number, data: number[]) {
  // Each drop is now a 4-vertex triangle strip (2 triangles) for tapered shape.
  // Vertices: 0=top-left, 1=top-right (tail), 2=bottom-left, 3=bottom-right (head)
  const vertsPerDrop = 4;
  const totalVerts = maxDrops * vertsPerDrop;
  const positions = new Float32Array(totalVerts * 3);
  const offsets = new Float32Array(totalVerts * 3);
  const speeds = new Float32Array(totalVerts);
  const phases = new Float32Array(totalVerts);
  const lengths = new Float32Array(totalVerts);
  const glows = new Float32Array(totalVerts);
  const hours = new Float32Array(totalVerts);
  const widths = new Float32Array(totalVerts);
  const vertexTs = new Float32Array(totalVerts);
  const indices: number[] = [];

  for (let i = 0; i < maxDrops; i++) {
    const hour = i % data.length;
    const x = -12.2 + (hour / (data.length - 1)) * 24.4 + (Math.random() - 0.5) * 0.46;
    const z = (Math.random() - 0.5) * 2.85 + Math.sin(i * 1.73) * 0.12;
    const speed = 0.20 + Math.random() * 0.18;
    const phase = Math.random();
    const length = 0.16 + Math.pow(Math.random(), 3.0) * 0.65;
    const glow = 0.12 + Math.pow(Math.random(), 4.0) * 0.70;
    const width = 0.008 + Math.random() * 0.016;

    const base = i * vertsPerDrop;
    // Triangle strip: top-left(0), top-right(1), bottom-left(2), bottom-right(3)
    // Positions in local strip space: x = side offset, y = along-drop (0=bottom/head, -1=top/tail)
    const localData = [
      { px: -0.5, py: -1, t: 1.0 },  // top-left (tail)
      { px:  0.5, py: -1, t: 1.0 },  // top-right (tail)
      { px: -0.5, py:  0, t: 0.0 },  // bottom-left (head)
      { px:  0.5, py:  0, t: 0.0 },  // bottom-right (head)
    ];

    for (let v = 0; v < vertsPerDrop; v++) {
      const idx = base + v;
      const p = idx * 3;
      positions[p] = localData[v].px;     // x: side offset for width
      positions[p + 1] = localData[v].py; // y: along-drop
      positions[p + 2] = 0;
      offsets[p] = x;
      offsets[p + 1] = 0;
      offsets[p + 2] = z;
      speeds[idx] = speed;
      phases[idx] = phase;
      lengths[idx] = length;
      glows[idx] = glow;
      hours[idx] = hour;
      widths[idx] = width;
      vertexTs[idx] = localData[v].t;
    }

    // Two triangles: (0,2,1), (1,2,3)
    indices.push(base, base + 2, base + 1);
    indices.push(base + 1, base + 2, base + 3);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aOffset", new THREE.BufferAttribute(offsets, 3));
  geometry.setAttribute("aSpeed", new THREE.BufferAttribute(speeds, 1));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  geometry.setAttribute("aLength", new THREE.BufferAttribute(lengths, 1));
  geometry.setAttribute("aGlow", new THREE.BufferAttribute(glows, 1));
  geometry.setAttribute("aHour", new THREE.BufferAttribute(hours, 1));
  geometry.setAttribute("aWidth", new THREE.BufferAttribute(widths, 1));
  geometry.setAttribute("aVertexT", new THREE.BufferAttribute(vertexTs, 1));
  geometry.setIndex(indices);
  return geometry;
}

function makeImpactGeometry(count: number, data: number[]) {
  const positions = new Float32Array(count * 3);
  const offsets = new Float32Array(count * 3);
  const phases = new Float32Array(count);
  const sizes = new Float32Array(count);
  const total = data.reduce((sum, value) => sum + Math.max(0.08, value), 0);

  for (let i = 0; i < count; i++) {
    const hour = weightedHour(data, total);
    offsets[i * 3] = -12.2 + (hour / (data.length - 1)) * 24.4 + (Math.random() - 0.5) * 0.8;
    offsets[i * 3 + 2] = (Math.random() - 0.5) * 4.5;
    phases[i] = Math.random();
    sizes[i] = 1.0 + Math.random() * 2.4;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aOffset", new THREE.BufferAttribute(offsets, 3));
  geometry.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  geometry.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  return geometry;
}

function makeTextSprite(
  text: string,
  options: { size?: number; opacity?: number; mono?: boolean } = {},
) {
  const size = options.size ?? 54;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d")!;
  const font = `${options.mono ? "400" : "500"} ${size}px ${
    options.mono ? "Consolas, monospace" : '"Microsoft YaHei", sans-serif'
  }`;
  context.font = font;
  const metrics = context.measureText(text);
  const padding = 18;
  canvas.width = Math.ceil(metrics.width + padding * 2);
  canvas.height = Math.ceil(size * 1.55);
  const draw = canvas.getContext("2d")!;
  draw.font = font;
  draw.textAlign = "center";
  draw.textBaseline = "middle";
  draw.fillStyle = `rgba(224, 231, 234, ${options.opacity ?? 0.56})`;
  draw.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
  });
  const sprite = new THREE.Sprite(material);
  const aspect = canvas.width / canvas.height;
  const height = size >= 50 ? 0.52 : 0.3;
  sprite.scale.set(height * aspect, height, 1);
  sprite.userData.texture = texture;
  return sprite;
}

function makeAxis() {
  const group = new THREE.Group();
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0xf1f7fa,
    transparent: false,
    opacity: 1,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const faintMaterial = new THREE.LineBasicMaterial({
    color: 0xd5e6ed,
    transparent: true,
    opacity: 0.78,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });

  const addLine = (points: THREE.Vector3[], material = lineMaterial) => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    group.add(new THREE.Line(geometry, material));
  };

  addLine([new THREE.Vector3(-13.1, 0, 2.7), new THREE.Vector3(13.4, 0, 2.7)]);
  addLine([new THREE.Vector3(-13.1, 0, 2.7), new THREE.Vector3(-13.1, 9.1, 2.7)]);

  for (let i = 0; i <= 6; i++) {
    const x = -12.2 + (i / 6) * 24.4;
    addLine([new THREE.Vector3(x, 0, 2.66), new THREE.Vector3(x, 0, 2.88)], faintMaterial);
    const time = makeTextSprite(`${String(i * 4).padStart(2, "0")}:00`, {
      size: 30,
      opacity: 1,
      mono: true,
    });
    time.position.set(x, -0.29, 3.02);
    group.add(time);
  }
  for (let i = 1; i <= 3; i++) {
    const y = i * 2.8;
    addLine([new THREE.Vector3(-13.1, y, 2.65), new THREE.Vector3(-12.85, y, 2.65)], faintMaterial);
    const value = makeTextSprite((i * 4.3).toFixed(1), {
      size: 28,
      opacity: 0.96,
      mono: true,
    });
    value.position.set(-13.52, y, 2.7);
    group.add(value);
  }

  const title = makeTextSprite("降雨强度", { size: 58, opacity: 1 });
  title.position.set(-13.15, 10.45, 2.65);
  title.scale.multiplyScalar(1.55);
  group.add(title);
  const subtitle = makeTextSprite("每小时降水", { size: 34, opacity: 0.96 });
  subtitle.position.set(-13.15, 9.68, 2.65);
  group.add(subtitle);
  const unit = makeTextSprite("毫米 / 小时", { size: 27, opacity: 0.86 });
  unit.position.set(-13.45, 8.82, 2.65);
  group.add(unit);

  return group;
}

function selectProfile() {
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;
  const cores = navigator.hardwareConcurrency ?? 4;
  const mobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  if (!mobile && memory >= 8 && cores >= 12) return { name: "ULTRA", drops: 52000, dpr: 1.55 };
  if (!mobile && memory >= 6 && cores >= 8) return { name: "HIGH", drops: 40000, dpr: 1.35 };
  if (memory >= 4 && cores >= 4) return { name: "BALANCED", drops: 28000, dpr: 1.15 };
  return { name: "ECO", drops: 14000, dpr: 1 };
}

type RainAudioController = {
  context: AudioContext;
  setLevel: (level: number) => void;
  stop: () => void;
};

function createRainAudio(): RainAudioController {
  const context = new AudioContext();
  const seconds = 5;
  const length = context.sampleRate * seconds;
  const master = context.createGain();
  master.gain.value = 0.0001;
  master.connect(context.destination);

  const makeSource = (kind: "white" | "brown" | "drops") => {
    const buffer = context.createBuffer(1, length, context.sampleRate);
    const channel = buffer.getChannelData(0);
    let brown = 0;
    for (let i = 0; i < length; i++) {
      if (kind === "white") {
        channel[i] = (Math.random() * 2 - 1) * 0.62;
      } else if (kind === "brown") {
        brown = (brown + 0.018 * (Math.random() * 2 - 1)) / 1.018;
        channel[i] = brown * 2.8;
      }
    }
    if (kind === "drops") {
      for (let drop = 0; drop < 70; drop++) {
        const start = Math.floor(Math.random() * (length - 2200));
        const decay = 260 + Math.floor(Math.random() * 1200);
        const pitch = 900 + Math.random() * 2300;
        for (let i = 0; i < decay; i++) {
          const envelope = Math.exp(-i / (decay * 0.18));
          channel[start + i] += Math.sin((i / context.sampleRate) * pitch * Math.PI * 2)
            * envelope * (0.16 + Math.random() * 0.05);
        }
      }
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.loop = true;
    return source;
  };

  const hiss = makeSource("white");
  const hissHigh = context.createBiquadFilter();
  hissHigh.type = "highpass";
  hissHigh.frequency.value = 950;
  const hissLow = context.createBiquadFilter();
  hissLow.type = "lowpass";
  hissLow.frequency.value = 7200;
  const hissGain = context.createGain();
  hissGain.gain.value = 0.72;
  hiss.connect(hissHigh).connect(hissLow).connect(hissGain).connect(master);

  const rumble = makeSource("brown");
  const rumbleFilter = context.createBiquadFilter();
  rumbleFilter.type = "lowpass";
  rumbleFilter.frequency.value = 420;
  const rumbleGain = context.createGain();
  rumbleGain.gain.value = 0.34;
  rumble.connect(rumbleFilter).connect(rumbleGain).connect(master);

  const drops = makeSource("drops");
  const dropFilter = context.createBiquadFilter();
  dropFilter.type = "bandpass";
  dropFilter.frequency.value = 2100;
  dropFilter.Q.value = 0.7;
  const dropGain = context.createGain();
  dropGain.gain.value = 0.52;
  drops.connect(dropFilter).connect(dropGain).connect(master);

  hiss.start();
  rumble.start();
  drops.start();
  const now = context.currentTime;
  master.gain.exponentialRampToValueAtTime(0.075, now + 0.8);

  return {
    context,
    setLevel(level: number) {
      const target = 0.032 + Math.min(1, Math.max(0, level)) * 0.065;
      master.gain.setTargetAtTime(target, context.currentTime, 0.35);
    },
    stop() {
      hiss.stop();
      rumble.stop();
      drops.stop();
      void context.close();
    },
  };
}

function RainCurveEditor({
  values,
  onChange,
}: {
  values: number[];
  onChange: (values: number[]) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activeRef = useRef<number | null>(null);
  const valuesRef = useRef(values);
  const maxValue = 14;

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    const padding = { left: 28, right: 12, top: 14, bottom: 23 };
    const plotWidth = rect.width - padding.left - padding.right;
    const plotHeight = rect.height - padding.top - padding.bottom;
    const xFor = (index: number) => padding.left + (index / (values.length - 1)) * plotWidth;
    const yFor = (value: number) => padding.top + (1 - value / maxValue) * plotHeight;

    context.strokeStyle = "rgba(203, 220, 226, 0.09)";
    context.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (i / 4) * plotHeight;
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(rect.width - padding.right, y);
      context.stroke();
    }
    for (let i = 0; i <= 6; i++) {
      const x = padding.left + (i / 6) * plotWidth;
      context.beginPath();
      context.moveTo(x, padding.top);
      context.lineTo(x, rect.height - padding.bottom);
      context.stroke();
    }
    context.fillStyle = "rgba(204, 220, 226, 0.36)";
    context.font = "7px Consolas, monospace";
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (let hour = 0; hour < 24; hour += 2) {
      context.fillText(String(hour).padStart(2, "0"), xFor(hour), rect.height - 8);
    }

    const gradient = context.createLinearGradient(0, padding.top, 0, rect.height - padding.bottom);
    gradient.addColorStop(0, "rgba(165, 205, 220, 0.15)");
    gradient.addColorStop(1, "rgba(165, 205, 220, 0)");
    context.beginPath();
    values.forEach((value, index) => {
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.lineTo(xFor(values.length - 1), rect.height - padding.bottom);
    context.lineTo(xFor(0), rect.height - padding.bottom);
    context.closePath();
    context.fillStyle = gradient;
    context.fill();

    context.beginPath();
    values.forEach((value, index) => {
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = "rgba(207, 224, 230, 0.7)";
    context.lineWidth = 1.35;
    context.stroke();

    values.forEach((value, index) => {
      const x = xFor(index);
      const y = yFor(value);
      context.beginPath();
      context.arc(x, y, activeRef.current === index ? 4.2 : 2.5, 0, Math.PI * 2);
      context.fillStyle = activeRef.current === index
        ? "rgba(227, 239, 243, 0.95)"
        : "rgba(180, 211, 223, 0.68)";
      context.fill();
    });

    context.textAlign = "right";
    context.fillText("14", padding.left - 7, padding.top + 3);
    context.fillText("7", padding.left - 7, padding.top + plotHeight / 2 + 3);
    context.fillText("0", padding.left - 7, rect.height - padding.bottom + 3);
  };

  useEffect(() => {
    valuesRef.current = values;
    draw();
    const observer = new ResizeObserver(draw);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [values]);

  const updateFromPointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const padding = { left: 28, right: 12, top: 14, bottom: 25 };
    const plotWidth = rect.width - padding.left - padding.right;
    const plotHeight = rect.height - padding.top - padding.bottom;
    const index = Math.max(
      0,
      Math.min(
        values.length - 1,
        Math.round(((event.clientX - rect.left - padding.left) / plotWidth) * (values.length - 1)),
      ),
    );
    const value = Math.max(
      0,
      Math.min(maxValue, (1 - (event.clientY - rect.top - padding.top) / plotHeight) * maxValue),
    );
    const next = [...valuesRef.current];
    next[index] = Math.round(value * 10) / 10;
    activeRef.current = index;
    valuesRef.current = next;
    onChange(next);
  };

  return (
    <canvas
      ref={canvasRef}
      className="curve-editor"
      aria-label="拖动曲线控制点编辑每小时降雨量"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        updateFromPointer(event);
      }}
      onPointerMove={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) updateFromPointer(event);
      }}
      onPointerUp={(event) => {
        updateFromPointer(event);
        event.currentTarget.releasePointerCapture(event.pointerId);
        activeRef.current = null;
      }}
      onPointerCancel={(event) => {
        activeRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
        draw();
      }}
    />
  );
}

export default function RainExperience() {
  const hostRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<RainAudioController | null>(null);
  const soundWantedRef = useRef(true);
  const axisGroupRef = useRef<THREE.Group | null>(null);
  const resetViewRef = useRef<(() => void) | null>(null);
  const updateRainfallRef = useRef<((data: number[]) => void) | null>(null);
  const rainfallRef = useRef(initialRainfall);
  const [rainfall, setRainfall] = useState(initialRainfall);
  const [editorOpen, setEditorOpen] = useState(false);
  const [numbersOpen, setNumbersOpen] = useState(false);
  const [soundOn, setSoundOn] = useState(false);
  const [axesVisible, setAxesVisible] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [hoveredHour, setHoveredHour] = useState<number | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry>({ fps: 60, drops: 0, profile: "检测中" });
  const average = useMemo(
    () => rainfall.reduce((sum, value) => sum + value, 0) / rainfall.length,
    [rainfall],
  );

  useEffect(() => {
    audioRef.current?.setLevel(average / 12);
  }, [average]);

  useEffect(() => {
    if (axisGroupRef.current) axisGroupRef.current.visible = axesVisible;
  }, [axesVisible]);

  useEffect(() => {
    return () => audioRef.current?.stop();
  }, []);

  useEffect(() => {
    soundWantedRef.current = window.localStorage.getItem("data-rain-sound") !== "off";
    const unlockAmbientRain = async (event: PointerEvent) => {
      if (!soundWantedRef.current) return;
      if ((event.target as Element | null)?.closest(".sound-trigger")) return;
      if (!audioRef.current) {
        audioRef.current = createRainAudio();
        const data = rainfallRef.current;
        audioRef.current.setLevel(data.reduce((sum, value) => sum + value, 0) / data.length / 12);
      }
      await audioRef.current.context.resume();
      setSoundOn(true);
    };
    window.addEventListener("pointerdown", unlockAmbientRain);
    return () => window.removeEventListener("pointerdown", unlockAmbientRain);
  }, []);

  const toggleSound = async () => {
    if (!audioRef.current) {
      audioRef.current = createRainAudio();
      audioRef.current.setLevel(average / 12);
      await audioRef.current.context.resume();
      soundWantedRef.current = true;
      window.localStorage.setItem("data-rain-sound", "on");
      setSoundOn(true);
      return;
    }
    if (audioRef.current.context.state === "running") {
      await audioRef.current.context.suspend();
      soundWantedRef.current = false;
      window.localStorage.setItem("data-rain-sound", "off");
      setSoundOn(false);
    } else {
      await audioRef.current.context.resume();
      soundWantedRef.current = true;
      window.localStorage.setItem("data-rain-sound", "on");
      setSoundOn(true);
    }
  };

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const profile = selectProfile();
    const data = rainfallRef.current;
    const maxDrops = 56000;
    let targetDrops = profile.drops;
    let currentDrops = profile.drops;
    let frameCount = 0;
    let sampleStart = performance.now();
    let raf = 0;
    let disposed = false;
    const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const pointer = new THREE.Vector2();
    const pointerSmooth = new THREE.Vector2();
    const mouseWorld = new THREE.Vector2(1000, 1000);
    const raycaster = new THREE.Raycaster();
    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const groundHit = new THREE.Vector3();
    let lastPointerMove = -100;
    let lastRainImpact = -1;
    let impactCursor = 0;

    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      powerPreference: "high-performance",
    });
    const dpr = Math.min(window.devicePixelRatio, profile.dpr);
    renderer.setPixelRatio(dpr);
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.setClearColor(0x000101, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.28;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000101, 0.038);
    const camera = new THREE.PerspectiveCamera(42, host.clientWidth / host.clientHeight, 0.1, 100);
    camera.position.set(15.8, 6.6, 17.8);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(-0.8, 2.2, -0.4);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.rotateSpeed = 0.48;
    controls.zoomSpeed = 0.72;
    controls.enablePan = false;
    controls.minDistance = 7.5;
    controls.maxDistance = 39;
    controls.minPolarAngle = 0.14;
    controls.maxPolarAngle = Math.PI / 2 - 0.035;
    controls.autoRotate = !reducedMotion;
    controls.autoRotateSpeed = 0.24;
    controls.saveState();
    const stopAutoRotate = () => {
      controls.autoRotate = false;
    };
    controls.addEventListener("start", stopAutoRotate);
    resetViewRef.current = () => {
      controls.reset();
      controls.autoRotate = !reducedMotion;
    };

    const composer = new EffectComposer(renderer);
    composer.setPixelRatio(dpr);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(
      new THREE.Vector2(host.clientWidth, host.clientHeight),
      0.14,
      0.28,
      0.91,
    );
    composer.addPass(bloom);

    // --- Lightning state ---
    let lightningIntensity = 0;
    let lightningTarget = 0;
    let nextLightningTime = 5 + Math.random() * 12;

    const rainUniforms = {
      uTime: { value: 0 },
      uRainfall: { value: new Float32Array(data) },
      uMouseWorld: { value: mouseWorld.clone() },
      uMouseForce: { value: 0 },
      uWind: { value: new THREE.Vector2() },
      uLightning: { value: 0 },
    };
    const rainGeometry = makeRainGeometry(maxDrops, data);
    // 6 indices per drop (2 triangles)
    rainGeometry.setDrawRange(0, currentDrops * 6);
    const rainMaterial = new THREE.ShaderMaterial({
      vertexShader: rainVertexShader,
      fragmentShader: rainFragmentShader,
      uniforms: rainUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(rainGeometry, rainMaterial));

    // --- Splash particle system ---
    const maxSplashParticles = 2000;
    const splashPositions = new Float32Array(maxSplashParticles * 3);
    const splashOrigins = new Float32Array(maxSplashParticles * 3);
    const splashVelocities = new Float32Array(maxSplashParticles * 3);
    const splashBirthTimes = new Float32Array(maxSplashParticles).fill(-100);
    const splashLifespans = new Float32Array(maxSplashParticles).fill(0.3);
    const splashSizes = new Float32Array(maxSplashParticles).fill(1);
    let splashCursor = 0;

    const splashGeometry = new THREE.BufferGeometry();
    splashGeometry.setAttribute("position", new THREE.BufferAttribute(splashPositions, 3));
    splashGeometry.setAttribute("aOrigin", new THREE.BufferAttribute(splashOrigins, 3));
    splashGeometry.setAttribute("aVelocity", new THREE.BufferAttribute(splashVelocities, 3));
    splashGeometry.setAttribute("aBirthTime", new THREE.BufferAttribute(splashBirthTimes, 1));
    splashGeometry.setAttribute("aLifespan", new THREE.BufferAttribute(splashLifespans, 1));
    splashGeometry.setAttribute("aSize", new THREE.BufferAttribute(splashSizes, 1));

    const splashUniforms = {
      uTime: { value: 0 },
      uPixelRatio: { value: dpr },
      uLightning: { value: 0 },
    };
    const splashMaterial = new THREE.ShaderMaterial({
      vertexShader: splashVertexShader,
      fragmentShader: splashFragmentShader,
      uniforms: splashUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const splashMesh = new THREE.Points(splashGeometry, splashMaterial);
    scene.add(splashMesh);

    const emitSplash = (x: number, z: number, time: number, intensity: number) => {
      const count = 3 + Math.floor(intensity * 5);
      for (let i = 0; i < count; i++) {
        const idx = splashCursor;
        splashCursor = (splashCursor + 1) % maxSplashParticles;
        const angle = Math.random() * Math.PI * 2;
        const speed = 0.3 + Math.random() * 0.8 * intensity;
        const p = idx * 3;
        splashOrigins[p] = x + (Math.random() - 0.5) * 0.05;
        splashOrigins[p + 1] = 0.02;
        splashOrigins[p + 2] = z + (Math.random() - 0.5) * 0.05;
        splashVelocities[p] = Math.cos(angle) * speed;
        splashVelocities[p + 1] = 1.2 + Math.random() * 1.5 * intensity;
        splashVelocities[p + 2] = Math.sin(angle) * speed;
        splashBirthTimes[idx] = time;
        splashLifespans[idx] = 0.15 + Math.random() * 0.25;
        splashSizes[idx] = 0.8 + Math.random() * 1.5;
      }
      (splashGeometry.attributes.aOrigin as THREE.BufferAttribute).needsUpdate = true;
      (splashGeometry.attributes.aVelocity as THREE.BufferAttribute).needsUpdate = true;
      (splashGeometry.attributes.aBirthTime as THREE.BufferAttribute).needsUpdate = true;
      (splashGeometry.attributes.aLifespan as THREE.BufferAttribute).needsUpdate = true;
      (splashGeometry.attributes.aSize as THREE.BufferAttribute).needsUpdate = true;
    };

    // --- Atmospheric fog layers ---
    const fogLayerMaterial = new THREE.ShaderMaterial({
      vertexShader: fogLayerVertexShader,
      fragmentShader: fogLayerFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uLightning: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
    });
    const fogLayers: THREE.Mesh[] = [];
    const fogPositions = [
      { y: 0.3, z: -3, scaleX: 52, scaleY: 2.8, rotX: -0.12 },
      { y: 1.5, z: 4, scaleX: 48, scaleY: 3.5, rotX: 0.08 },
      { y: 0.1, z: -8, scaleX: 55, scaleY: 2.2, rotX: -0.05 },
    ];
    for (const fp of fogPositions) {
      const fog = new THREE.Mesh(
        new THREE.PlaneGeometry(1, 1),
        fogLayerMaterial,
      );
      fog.position.set(0, fp.y, fp.z);
      fog.scale.set(fp.scaleX, fp.scaleY, 1);
      fog.rotation.x = fp.rotX;
      fog.renderOrder = 2;
      scene.add(fog);
      fogLayers.push(fog);
    }

    // --- Sky dome: dark gradient hemisphere ---
    const skyGeometry = new THREE.SphereGeometry(45, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
    const skyMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPos.xyz;
          gl_Position = projectionMatrix * viewMatrix * worldPos;
        }
      `,
      fragmentShader: `
        uniform float uLightning;
        varying vec3 vWorldPosition;
        void main() {
          float h = normalize(vWorldPosition).y;
          vec3 low = vec3(0.003, 0.006, 0.010);
          vec3 high = vec3(0.015, 0.022, 0.032);
          vec3 color = mix(low, high, smoothstep(0.0, 0.6, h));
          // Lightning flash in sky
          color += vec3(0.04, 0.055, 0.07) * uLightning * smoothstep(0.1, 0.5, h);
          gl_FragColor = vec4(color, 1.0);
        }
      `,
      uniforms: { uLightning: { value: 0 } },
      side: THREE.BackSide,
      depthWrite: false,
    });
    const sky = new THREE.Mesh(skyGeometry, skyMaterial);
    sky.renderOrder = -1;
    scene.add(sky);

    const waterWidth = profile.name === "ECO" ? 128 : 256;
    const waterHeight = profile.name === "ECO" ? 64 : 128;
    const gpuCompute = new GPUComputationRenderer(waterWidth, waterHeight, renderer);
    const initialWater = gpuCompute.createTexture();
    initialWater.image.data!.fill(0);
    const heightmapVariable = gpuCompute.addVariable("heightmap", waterComputeShader, initialWater);
    gpuCompute.setVariableDependencies(heightmapVariable, [heightmapVariable]);
    const computeUniforms = heightmapVariable.material.uniforms;
    computeUniforms.uTime = { value: 0 };
    const waterImpacts = Array.from(
      { length: 16 },
      () => new THREE.Vector4(1000, 1000, -1, 0),
    );
    computeUniforms.uImpacts = { value: waterImpacts };
    const computeError = gpuCompute.init();
    if (computeError) console.error(computeError);

    const floorUniforms = {
      heightmap: { value: gpuCompute.getCurrentRenderTarget(heightmapVariable).texture },
      uTexel: { value: new THREE.Vector2(1 / waterWidth, 1 / waterHeight) },
      uLightning: { value: 0 },
    };
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(48, 22, waterWidth - 1, waterHeight - 1),
      new THREE.ShaderMaterial({
        vertexShader: floorVertexShader,
        fragmentShader: floorFragmentShader,
        uniforms: floorUniforms,
      }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.035;
    scene.add(floor);

    let rainData = data;
    let rainRate = 0;
    let rainTotal = 0;
    const updateRainfall = (next: number[]) => {
      rainData = next;
      // Only upload 24 values; keep particle buffers, camera and water history alive.
      rainUniforms.uRainfall.value.set(next);
      const mean = next.reduce((sum, value) => sum + value, 0) / next.length;
      rainRate = 1.7 + mean * 0.28;
      rainTotal = next.reduce((sum, value) => sum + Math.max(0.08, value), 0);
    };
    updateRainfallRef.current = updateRainfall;
    updateRainfall(data);

    const axis = makeAxis();
    axis.visible = false;
    axisGroupRef.current = axis;
    scene.add(axis);

    const updatePointerWorld = (event: PointerEvent) => {
      const rect = host.getBoundingClientRect();
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      if (raycaster.ray.intersectPlane(groundPlane, groundHit)) {
        mouseWorld.set(groundHit.x, groundHit.z);
        rainUniforms.uMouseWorld.value.copy(mouseWorld);
        if (
          mouseWorld.x >= -12.7 &&
          mouseWorld.x <= 12.7 &&
          Math.abs(mouseWorld.y) <= 3.25
        ) {
          const hour = Math.max(
            0,
            Math.min(
              rainData.length - 1,
              Math.round(((mouseWorld.x + 12.2) / 24.4) * (rainData.length - 1)),
            ),
          );
          setHoveredHour(hour);
        } else {
          setHoveredHour(null);
        }
      }
    };
    const onPointer = (event: PointerEvent) => {
      updatePointerWorld(event);
      lastPointerMove = clock.getElapsedTime();
    };
    const onPointerDown = (event: PointerEvent) => {
      updatePointerWorld(event);
      const elapsed = clock.getElapsedTime();
      waterImpacts[impactCursor].set(mouseWorld.x, mouseWorld.y, elapsed, 0.034);
      impactCursor = (impactCursor + 1) % waterImpacts.length;
    };
    const onResize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      composer.setSize(width, height);
    };
    host.addEventListener("pointermove", onPointer);
    host.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("resize", onResize);

    const clock = new THREE.Clock();
    const animate = () => {
      if (disposed) return;
      raf = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      pointerSmooth.lerp(pointer, 0.025);
      rainUniforms.uTime.value = elapsed;
      const mouseForce = Math.max(0, 1 - (elapsed - lastPointerMove) * 0.65);
      rainUniforms.uMouseForce.value += (mouseForce - rainUniforms.uMouseForce.value) * 0.08;
      rainUniforms.uWind.value.set(
        pointerSmooth.x * 0.34 + Math.sin(elapsed * 0.17) * 0.035,
        pointerSmooth.y * 0.12,
      );

      // --- Lightning update ---
      if (elapsed > nextLightningTime) {
        lightningTarget = 0.6 + Math.random() * 0.4;
        // Schedule double-flash occasionally
        const doubleFlash = Math.random() < 0.3;
        nextLightningTime = elapsed + (doubleFlash ? 0.08 : (4 + Math.random() * 14));
        if (doubleFlash) lightningTarget *= 0.7;
      }
      lightningIntensity += (lightningTarget - lightningIntensity) * 0.3;
      lightningTarget *= 0.88; // rapid decay
      if (lightningTarget < 0.01) lightningTarget = 0;
      rainUniforms.uLightning.value = lightningIntensity;
      splashUniforms.uLightning.value = lightningIntensity;
      splashUniforms.uTime.value = elapsed;
      floorUniforms.uLightning.value = lightningIntensity;
      fogLayerMaterial.uniforms.uTime.value = elapsed;
      fogLayerMaterial.uniforms.uLightning.value = lightningIntensity;
      (skyMaterial.uniforms.uLightning as { value: number }).value = lightningIntensity;
      // Adjust tone mapping exposure during lightning
      renderer.toneMappingExposure = 0.28 + lightningIntensity * 0.15;

      while (elapsed - lastRainImpact > 1 / rainRate) {
        lastRainImpact += 1 / rainRate;
        const hour = weightedHour(rainData, rainTotal);
        const x = -12.2 + (hour / (rainData.length - 1)) * 24.4 + (Math.random() - 0.5) * 0.5;
        const z = (Math.random() - 0.5) * 20.0;
        const strength = 0.009 + Math.min(1.2, rainData[hour] / 10) * 0.012;
        waterImpacts[impactCursor].set(x, z, lastRainImpact, strength);
        impactCursor = (impactCursor + 1) % waterImpacts.length;
        // Emit splash particles at impact point
        const splashIntensity = Math.min(1, rainData[hour] / 8);
        if (Math.abs(z) < 11 && Math.abs(x) < 13) {
          emitSplash(x, z, lastRainImpact, splashIntensity);
        }
      }
      computeUniforms.uTime.value = elapsed;
      gpuCompute.compute();
      floorUniforms.heightmap.value = gpuCompute.getCurrentRenderTarget(heightmapVariable).texture;

      currentDrops += (targetDrops - currentDrops) * 0.025;
      // 6 indices per drop (2 triangles × 3 verts)
      rainGeometry.setDrawRange(0, Math.floor(currentDrops) * 6);

      controls.update();

      frameCount++;
      const now = performance.now();
      if (now - sampleStart > 1900) {
        const fps = Math.round((frameCount * 1000) / (now - sampleStart));
        if (fps < 43) targetDrops = Math.max(10000, targetDrops * 0.82);
        if (fps > 57) targetDrops = Math.min(maxDrops, targetDrops * 1.07);
        setTelemetry({
          fps,
          drops: Math.round(currentDrops / 1000) * 1000,
          profile: profile.name,
        });
        frameCount = 0;
        sampleStart = now;
      }
      composer.render();
    };

    setTelemetry({ fps: 60, drops: profile.drops, profile: profile.name });
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      host.removeEventListener("pointermove", onPointer);
      host.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("resize", onResize);
      controls.removeEventListener("start", stopAutoRotate);
      controls.dispose();
      resetViewRef.current = null;
      updateRainfallRef.current = null;
      axisGroupRef.current = null;
      rainGeometry.dispose();
      rainMaterial.dispose();
      splashGeometry.dispose();
      splashMaterial.dispose();
      scene.remove(splashMesh);
      fogLayers.forEach((f) => { scene.remove(f); f.geometry.dispose(); });
      fogLayerMaterial.dispose();
      scene.remove(sky);
      skyGeometry.dispose();
      skyMaterial.dispose();
      gpuCompute.dispose();
      floor.geometry.dispose();
      (floor.material as THREE.Material).dispose();
      axis.traverse((child) => {
        if (child instanceof THREE.Line) {
          child.geometry.dispose();
          child.material.dispose();
        }
        if (child instanceof THREE.Sprite) {
          child.material.dispose();
          const texture = child.userData.texture as THREE.Texture | undefined;
          texture?.dispose();
        }
      });
      composer.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    rainfallRef.current = rainfall;
    updateRainfallRef.current?.(rainfall);
  }, [rainfall]);

  return (
    <main className="rain-app">
      <div ref={hostRef} className="rain-canvas" aria-hidden="true" />
      <div className="vignette" aria-hidden="true" />

      <button
        type="button"
        className="control-reveal"
        onClick={() => setControlsVisible((value) => !value)}
        aria-label={controlsVisible ? "隐藏控制" : "显示控制"}
        aria-pressed={controlsVisible}
      >
        {controlsVisible ? "−" : "+"}
      </button>
      <div className={`scene-actions ${controlsVisible ? "" : "hidden"}`}>
        <button
          type="button"
          className={`sound-trigger ${soundOn ? "active" : ""}`}
          onClick={toggleSound}
          aria-pressed={soundOn}
          aria-label="雨声"
          title="雨声"
        >
          ♫
        </button>
        <button
          type="button"
          className="axis-trigger"
          onClick={() => setAxesVisible((value) => !value)}
          aria-pressed={axesVisible}
          aria-label="坐标轴"
          title="坐标轴"
        >
          ⌗
        </button>
        <button type="button" className="reset-trigger" onClick={() => resetViewRef.current?.()} aria-label="复位视角" title="复位视角">
          ↺
        </button>
        <button
          type="button"
          className="edit-trigger"
          onClick={() => setEditorOpen(true)}
          aria-expanded={editorOpen}
          aria-label="编辑降雨"
          title="编辑降雨"
        >
          ≋
        </button>
      </div>

      <aside className={`data-editor ${editorOpen ? "open" : ""}`} aria-hidden={!editorOpen}>
        <div className="editor-head">
          <div>
            <small>00:00—24:00</small>
            <h2>编辑降雨数据</h2>
          </div>
          <button type="button" onClick={() => setEditorOpen(false)} aria-label="关闭编辑器">×</button>
        </div>
        <p>直接拖动曲线上的控制点，雨幕高度和水面落点会随拖动实时变化。</p>
        <RainCurveEditor
          values={rainfall}
          onChange={setRainfall}
        />
        <div className="curve-stats">
          <span>峰值 <b>{Math.max(...rainfall).toFixed(1)}</b> mm/h</span>
          <span>均值 <b>{average.toFixed(1)}</b> mm/h</span>
        </div>
        <button
          type="button"
          className="numbers-toggle"
          onClick={() => setNumbersOpen((value) => !value)}
          aria-expanded={numbersOpen}
        >
          {numbersOpen ? "收起数值" : "数字编辑"}
        </button>
        {numbersOpen && (
          <div className="data-grid" aria-label="逐小时降雨数值">
            {rainfall.map((value, index) => (
              <label key={index}>
                <span>{String(index).padStart(2, "0")}:00</span>
                <input
                  type="number"
                  min="0"
                  max="14"
                  step="0.1"
                  value={value}
                  onChange={(event) => {
                    const next = [...rainfall];
                    next[index] = Math.min(14, Math.max(0, Number(event.target.value)));
                    setRainfall(next);
                  }}
                  aria-label={`${String(index).padStart(2, "0")}:00 降雨量`}
                />
              </label>
            ))}
          </div>
        )}
        <button
          type="button"
          className="apply-data"
          onClick={() => setRainfall(initialRainfall)}
        >
          恢复示例曲线
        </button>
      </aside>
    </main>
  );
}
