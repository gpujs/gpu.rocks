// content/index.js — course content auto-discovery. THIS FILE IS NEVER EDITED
// AGAIN: module files self-register simply by existing as
// content/trackN/module-N-M.js with a default export matching the schema.

import trackMeta from './tracks';

const moduleFiles = import.meta.glob('./track*/module-*.js', { eager: true });

function idParts(id) {
  return String(id).split('-').map(Number);
}

// All modules, sorted by id ('1-2' before '1-10' before '2-1').
export const modules = Object.values(moduleFiles)
  .map(file => file.default)
  .filter(Boolean)
  .sort((a, b) => {
    const [aTrack, aNum] = idParts(a.id);
    const [bTrack, bNum] = idParts(b.id);
    return aTrack - bTrack || aNum - bNum;
  });

// Tracks (mockup metadata) with their modules attached.
export const tracks = trackMeta.map(track => ({
  ...track,
  modules: modules.filter(module => module.track === track.number),
}));

export function getModule(moduleId) {
  return modules.find(module => module.id === moduleId) || null;
}

// Task id: `${moduleId}-${taskNum}` with taskNum 1-based, e.g. '1-2-3'.
export function taskId(moduleId, taskNum) {
  return `${moduleId}-${taskNum}`;
}

// getTask('1-2', 3) → { module, task, taskNum, taskIndex, taskId, total }
// or null when the module or task does not exist.
export function getTask(moduleId, taskNum) {
  const module = getModule(moduleId);
  if (!module) return null;
  const num = Number(taskNum);
  if (!Number.isInteger(num) || num < 1 || num > module.tasks.length) return null;
  return {
    module,
    task: module.tasks[num - 1],
    taskNum: num,
    taskIndex: num - 1,
    taskId: taskId(moduleId, num),
    total: module.tasks.length,
  };
}
