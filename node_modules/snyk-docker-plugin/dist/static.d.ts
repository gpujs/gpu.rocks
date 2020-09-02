import { DockerFileAnalysis } from "./docker-file";
import { PluginResponse } from "./types";
export declare function analyzeStatically(targetImage: string, dockerfileAnalysis: DockerFileAnalysis | undefined, options: any): Promise<PluginResponse>;
export declare function isRequestingStaticAnalysis(options?: any): boolean;
