import { DockerFileAnalysis } from "./docker-file";
import * as types from "./types";
export { buildResponse };
declare function buildResponse(runtime: string | undefined, depsAnalysis: any, dockerfileAnalysis: DockerFileAnalysis | undefined, manifestFiles: types.ManifestFile[], options: any): types.PluginResponse;
