import * as THREE from 'three';

import splatMatFragSrc from './splatmat.frag.glsl?raw'
import { atomicAdd, atomicLoad, atomicStore, float, Fn, instancedArray, instanceIndex, radians, texture, textureStore, uint, uvec2, vec4, vec2, cos, sin, atan, If } from 'three/tsl';
import { MeshBasicNodeMaterial, StorageTexture, WebGPURenderer } from 'three/webgpu';

// -------------- TUNABLES -----------------------------
const WIDTH = 1280;
const HEIGHT = 720;
const STEP_SIZE = 5;
const TURN_RAD = 10;
const DECAY_RATE = 0.05;
const SENSOR_DIST = 10;
const SENSOR_ANGLE = 25;
const W = 0.2;
const DROP = 20;

const count = 360000;

const renderer = new WebGPURenderer();
await renderer.init();

renderer.setSize(WIDTH, HEIGHT);
document.body.appendChild(renderer.domElement);

// -------------- SIMULATION ---------------------------
const degToRad = degrees => degrees * (Math.PI / 180);

const startBuffer = new Float32Array(4 * count);
for (let i = 0; i < count; i++) {
  startBuffer[i * 4 + 0] = Math.random() * WIDTH;
  startBuffer[i * 4 + 1] = Math.random() * HEIGHT;
  const a = Math.random() * 360;
  startBuffer[i * 4 + 2] = Math.cos(degToRad(a));
  startBuffer[i * 4 + 3] = Math.sin(degToRad(a));
}

const agentBuffer = instancedArray(startBuffer, 'vec4');
const trailMap = instancedArray(WIDTH * HEIGHT, 'uint').toAtomic();
const trailBuffer = instancedArray(WIDTH * HEIGHT, 'float');

const trailMapTexture = new StorageTexture(WIDTH, HEIGHT);
trailMapTexture.magFilter = THREE.NearestFilter;
trailMapTexture.minFilter = THREE.NearestFilter;

const createAgentNode = Fn(() => {
  const currentAgent = agentBuffer.element(instanceIndex);
  currentAgent.assign(vec4(WIDTH/2, HEIGHT/2, cos(radians(instanceIndex.toFloat().mul(0.001))), sin(radians(instanceIndex.toFloat().mul(0.001)))));
})().compute(count);


const agentNode = Fn(() => {
  const currentAgent = agentBuffer.element(instanceIndex);

  const posX = currentAgent.x;
  const posY = currentAgent.y;

  const dirX = currentAgent.z;
  const dirY = currentAgent.w;



  let final_angle = float(0).toVar();
  const dir_angle = atan(dirY, dirX).toVar();
  const left_sensor_angle = dir_angle.sub(degToRad(SENSOR_ANGLE));
  const right_sensor_angle = dir_angle.add(degToRad(SENSOR_ANGLE));

  const left_sensor_dir = vec2(cos(left_sensor_angle), sin(left_sensor_angle));
  const right_sensor_dir = vec2(cos(right_sensor_angle), sin(right_sensor_angle));

  const front_sensor = posY.add(dirY.mul(SENSOR_DIST)).add(HEIGHT).mod(HEIGHT).floor().mul(WIDTH).add(posX.add(dirX.mul(SENSOR_DIST)).add(WIDTH).mod(WIDTH).floor()); // (y + dirY * SENSOR_DIST) * WIDTH + (posX + dirX * SENSOR_DIST)
  const left_sensor = posY.add(left_sensor_dir.y.mul(SENSOR_DIST)).add(HEIGHT).mod(HEIGHT).floor().mul(WIDTH).add(posX.add(left_sensor_dir.x.mul(SENSOR_DIST)).add(WIDTH).mod(WIDTH).floor()); // (y + sin(left_sensor_angle) * SENSOR_DIST) * WIDTH + (posX + cos(left_sensor_angle) * SENSOR_DIST)
  const right_sensor = posY.add(right_sensor_dir.y.mul(SENSOR_DIST)).add(HEIGHT).mod(HEIGHT).floor().mul(WIDTH).add(posX.add(right_sensor_dir.x.mul(SENSOR_DIST)).add(WIDTH).mod(WIDTH).floor()); // (y + sin(right_sensor_angle) * SENSOR_DIST) * WIDTH + (posX + cos(right_sensor_angle) * SENSOR_DIST)


  const front_sensor_value = trailBuffer.element(front_sensor).toVar();
  const left_sensor_value = trailBuffer.element(left_sensor).toVar();
  const right_sensor_value = trailBuffer.element(right_sensor).toVar();

  If(left_sensor_value.greaterThan(front_sensor_value).and(left_sensor_value.greaterThan(right_sensor_value)),
    () => { final_angle.assign(dir_angle.sub(degToRad(TURN_RAD))) }).
    ElseIf(right_sensor_value.greaterThan(front_sensor_value).and(right_sensor_value.greaterThan(left_sensor_value)),
      () => { final_angle.assign(dir_angle.add(degToRad(TURN_RAD))) }).
    Else(() => { final_angle.assign(dir_angle) });

  const newPos = vec4(
    posX.add(cos(final_angle).mul(STEP_SIZE)).add(WIDTH).mod(WIDTH),
    posY.add(sin(final_angle).mul(STEP_SIZE)).add(HEIGHT).mod(HEIGHT),
    cos(final_angle),
    sin(final_angle)
  );

  currentAgent.assign(newPos);

  atomicAdd(trailMap.element(posY.floor().mul(WIDTH).add(posX.floor()).toUint()), uint(1));
})().compute(count);

const trailMapTextureNode = Fn(() => {
  const x = instanceIndex.mod(WIDTH).toInt();
  const y = instanceIndex.div(WIDTH).toInt();

  let sum = float(0).toVar();
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const nx = x.add(dx).add(WIDTH).mod(WIDTH);
      const ny = y.add(dy).add(HEIGHT).mod(HEIGHT);
      const nIDX = ny.mul(WIDTH).add(nx).toUint();
      sum.addAssign(float(trailBuffer.element(nIDX)));
    }
  }

  const prev = trailBuffer.element(instanceIndex);
  const v = sum.div(9).mul(float(1 - DECAY_RATE)).add(float(atomicLoad(trailMap.element(instanceIndex)))).toVar();
  prev.assign(float(1-W).mul(prev).add(v.mul(W)));

  const out = v.div(DROP).toVar();
  textureStore(trailMapTexture, uvec2(x.toUint(), y.toUint()), vec4(out, out, out, 1)).toWriteOnly();
})().compute(WIDTH * HEIGHT);


const clearTrailMapNode = Fn(() => {
  atomicStore(trailMap.element(instanceIndex), uint(0));
})().compute(WIDTH * HEIGHT);


// -------------- CAMERA AND SCENE ----------------------
const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const plane = new THREE.PlaneGeometry(2, 2);
const material = new MeshBasicNodeMaterial();
material.colorNode = texture(trailMapTexture);// color('black');
const mesh = new THREE.Mesh(plane, material);
scene.add(mesh);
mesh.frustumCulled = false;


//await renderer.computeAsync(createAgentNode);

renderer.setAnimationLoop((now) => {
  //renderer.compute(agentNode, [count, 1, 1]);
  //
  renderer.compute(agentNode);
  renderer.compute(trailMapTextureNode);
  renderer.compute(clearTrailMapNode);
  renderer.render(scene, camera);
});
