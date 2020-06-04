import * as dockerFile from "./docker-file";
import { DepTree, PluginResponse, ScannedProjectExtended, ScannedProjectManifestFiles, ScanType } from "./types";
export { inspect, dockerFile, PluginResponse, ScannedProjectExtended, ScanType, ScannedProjectManifestFiles, DepTree, };
declare function inspect(root: string, targetFile?: string, options?: any): Promise<PluginResponse>;
