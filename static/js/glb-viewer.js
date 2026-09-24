import * as THREE from './three/three.module.min.js';
import { OrbitControls } from './three/OrbitControls.js';

/**
 * ScanNet++ exports are mostly glTF POINTS (mode 0) with optional camera meshes.
 * We only unpack the point cloud so display does not depend on texture / mesh loaders.
 */

const POINT_PIXEL_SIZE = 2.5;

function parseGlb(buffer) {
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) {
    throw new Error('Not a GLB file');
  }
  let offset = 12;
  let json = null;
  let bin = null;
  while (offset + 8 <= u8.byteLength) {
    const chunkLen = dv.getUint32(offset, true);
    const chunkType = dv.getUint32(offset + 4, true);
    const start = offset + 8;
    const end = start + chunkLen;
    if (chunkType === 0x4e4f534a) {
      json = JSON.parse(new TextDecoder().decode(u8.subarray(start, end)));
    } else if (chunkType === 0x004e4942) {
      bin = u8.subarray(start, end);
    }
    offset = end;
  }
  if (!json || !bin) throw new Error('Incomplete GLB');
  return { json, bin };
}

function readAccessor(json, bin, accessorIndex) {
  const acc = json.accessors[accessorIndex];
  const view = json.bufferViews[acc.bufferView];
  const byteOffset = (view.byteOffset || 0) + (acc.byteOffset || 0);
  const count = acc.count;
  const typeCount = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[acc.type];
  const componentType = acc.componentType;
  let TypedArray;
  let itemBytes;
  if (componentType === 5126) {
    TypedArray = Float32Array;
    itemBytes = 4;
  } else if (componentType === 5121) {
    TypedArray = Uint8Array;
    itemBytes = 1;
  } else if (componentType === 5123) {
    TypedArray = Uint16Array;
    itemBytes = 2;
  } else {
    throw new Error(`Unsupported componentType ${componentType}`);
  }
  const stride = view.byteStride || typeCount * itemBytes;
  const out = new TypedArray(count * typeCount);
  if (stride === typeCount * itemBytes) {
    out.set(
      new TypedArray(
        bin.buffer,
        bin.byteOffset + byteOffset,
        count * typeCount
      )
    );
  } else {
    for (let i = 0; i < count; i++) {
      const src = new TypedArray(
        bin.buffer,
        bin.byteOffset + byteOffset + i * stride,
        typeCount
      );
      out.set(src, i * typeCount);
    }
  }
  return { array: out, count, typeCount, normalized: !!acc.normalized, componentType };
}

function mat4FromNode(node) {
  if (node.matrix && node.matrix.length === 16) {
    return new THREE.Matrix4().fromArray(node.matrix);
  }
  const m = new THREE.Matrix4();
  const t = node.translation || [0, 0, 0];
  const r = node.rotation || [0, 0, 0, 1];
  const s = node.scale || [1, 1, 1];
  m.compose(
    new THREE.Vector3(t[0], t[1], t[2]),
    new THREE.Quaternion(r[0], r[1], r[2], r[3]),
    new THREE.Vector3(s[0], s[1], s[2])
  );
  return m;
}

function buildPointCloud(json, bin) {
  const group = new THREE.Group();
  const meshes = json.meshes || [];
  const nodes = json.nodes || [];

  nodes.forEach((node) => {
    if (node.mesh === undefined) return;
    const mesh = meshes[node.mesh];
    if (!mesh) return;
    const world = mat4FromNode(node);

    (mesh.primitives || []).forEach((prim) => {
      // 0 = POINTS
      if ((prim.mode ?? 4) !== 0) return;
      if (prim.attributes.POSITION === undefined) return;

      const pos = readAccessor(json, bin, prim.attributes.POSITION);
      const positions = new Float32Array(pos.array);

      // Apply node transform
      const v = new THREE.Vector3();
      for (let i = 0; i < pos.count; i++) {
        v.fromArray(positions, i * 3).applyMatrix4(world);
        positions[i * 3] = v.x;
        positions[i * 3 + 1] = v.y;
        positions[i * 3 + 2] = v.z;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      let hasColor = false;
      if (prim.attributes.COLOR_0 !== undefined) {
        const col = readAccessor(json, bin, prim.attributes.COLOR_0);
        let colors;
        if (col.componentType === 5121) {
          colors = new Float32Array(col.count * 3);
          for (let i = 0; i < col.count; i++) {
            const a = i * col.typeCount;
            const o = i * 3;
            const scale = col.normalized ? 1 / 255 : 1;
            colors[o] = col.array[a] * scale;
            colors[o + 1] = col.array[a + 1] * scale;
            colors[o + 2] = col.array[a + 2] * scale;
          }
        } else {
          colors = new Float32Array(col.count * 3);
          for (let i = 0; i < col.count; i++) {
            const a = i * col.typeCount;
            const o = i * 3;
            colors[o] = col.array[a];
            colors[o + 1] = col.array[a + 1];
            colors[o + 2] = col.array[a + 2];
          }
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        hasColor = true;
      }

      geometry.computeBoundingSphere();

      const material = new THREE.PointsMaterial({
        size: POINT_PIXEL_SIZE,
        sizeAttenuation: false,
        vertexColors: hasColor,
        color: hasColor ? 0xffffff : 0x5a8ab8,
        depthWrite: true,
      });

      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false;
      group.add(points);
    });
  });

  if (!group.children.length) {
    throw new Error('No point cloud (mode 0) found in GLB');
  }
  return group;
}

function fitCameraToObject(camera, controls, object, options = {}) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  // These exports put the reference camera near the origin looking +Z
  // (Y-up after the glTF node transform). Start from that frontal view.
  const fov = (camera.fov * Math.PI) / 180;
  const fitH = size.y > 1e-6 ? size.y / 2 / Math.tan(fov / 2) : 1;
  const fitW = size.x > 1e-6 ? size.x / 2 / (Math.tan(fov / 2) * camera.aspect) : 1;
  const dist = Math.max(fitH, fitW, size.z * 0.55, 0.8) * (options.distanceScale ?? 1.25);

  // Optional fine-tune via data attributes (degrees / multipliers)
  const yaw = ((options.yawDeg ?? 0) * Math.PI) / 180;
  const pitch = ((options.pitchDeg ?? 8) * Math.PI) / 180;

  // Base offset: stand on the -Z side of the cloud, look toward +Z
  const offset = new THREE.Vector3(
    Math.sin(yaw) * Math.cos(pitch),
    Math.sin(pitch),
    -Math.cos(yaw) * Math.cos(pitch)
  ).multiplyScalar(dist);

  controls.target.copy(center);
  camera.position.copy(center).add(offset);
  camera.up.set(0, 1, 0);

  // Keep orbit from jumping through the cloud (browser wheel is often more sensitive
  // than the IDE embedded preview).
  const radius = Math.max(size.length() * 0.5, 0.5);
  controls.minDistance = Math.max(radius * 0.15, 0.2);
  controls.maxDistance = Math.max(radius * 12, dist * 4);
  controls.zoomSpeed = 0.6;

  updateCameraClipPlanes(camera, controls);
  controls.update();
}

/** Keep near/far in sync with zoom distance so close-up dolly does not clip the cloud. */
function updateCameraClipPlanes(camera, controls) {
  const dist = Math.max(camera.position.distanceTo(controls.target), 0.05);
  camera.near = Math.max(dist / 200, 0.001);
  camera.far = Math.max(dist * 100, 200);
  camera.updateProjectionMatrix();
}

async function loadPointCloud(src) {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${src}`);
  const buffer = await res.arrayBuffer();
  const { json, bin } = parseGlb(buffer);
  return buildPointCloud(json, bin);
}

function initViewer(container) {
  const src = container.dataset.src;
  if (!src || container.dataset.booted === '1') return;
  container.dataset.booted = '1';

  const status = document.createElement('div');
  status.className = 'glb-status';
  status.textContent = 'Loading…';
  container.appendChild(status);

  const width = Math.max(container.clientWidth, 320);
  const height = Math.max(container.clientHeight, 280);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef1f4);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 2000);
  camera.position.set(2, 1.5, 3);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
  if ('outputColorSpace' in renderer) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  }
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.addEventListener('change', () => updateCameraClipPlanes(camera, controls));

  // Prevent page scroll from stealing wheel while hovering the canvas (external browsers).
  renderer.domElement.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
    },
    { passive: false }
  );

  function resize() {
    const w = Math.max(container.clientWidth, 1);
    const h = Math.max(container.clientHeight, 1);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  const ro = new ResizeObserver(resize);
  ro.observe(container);
  container.addEventListener('glb-resize', resize);

  let raf = 0;
  function tick() {
    raf = requestAnimationFrame(tick);
    if (container.offsetParent === null || container.clientWidth <= 0) return;
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  loadPointCloud(src)
    .then((root) => {
      scene.add(root);
      fitCameraToObject(camera, controls, root, {
        yawDeg: Number(container.dataset.yaw || 0),
        pitchDeg: Number(container.dataset.pitch || 8),
        distanceScale: Number(container.dataset.distance || 1.25),
      });
      status.remove();
      container.classList.add('is-loaded');
      requestAnimationFrame(() => {
        resize();
        renderer.render(scene, camera);
      });
    })
    .catch((err) => {
      console.error('Failed to load GLB', src, err);
      status.textContent = 'Failed to load 3D model.';
      status.classList.add('is-error');
      container.classList.add('is-error');
    });

  container._glbDispose = () => {
    cancelAnimationFrame(raf);
    ro.disconnect();
    controls.dispose();
    renderer.dispose();
  };
}

function bootVisible() {
  document.querySelectorAll('.glb-slide.active .glb-canvas[data-src]').forEach((el) => {
    if (el.dataset.booted === '1') {
      el.dispatchEvent(new Event('glb-resize'));
      return;
    }
    initViewer(el);
  });
}

function observeSection() {
  const section = document.getElementById('reconstruction');
  const tryBoot = () => bootVisible();

  if (!section) {
    tryBoot();
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) tryBoot();
      });
    },
    { rootMargin: '200px', threshold: 0.01 }
  );
  io.observe(section);

  window.addEventListener('hashchange', () => {
    if (location.hash === '#reconstruction') tryBoot();
  });

  // Nav clicks / already visible
  tryBoot();
  if (location.hash === '#reconstruction') {
    requestAnimationFrame(tryBoot);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', observeSection);
} else {
  observeSection();
}

window.bootGlbViewers = bootVisible;
