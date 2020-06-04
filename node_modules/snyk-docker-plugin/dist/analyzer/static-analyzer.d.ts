import { DockerFileAnalysis } from "../docker-file";
import { StaticAnalysisOptions } from "../types";
import { StaticAnalysis } from "./types";
export declare function analyze(targetImage: string, dockerfileAnalysis: DockerFileAnalysis | undefined, options: StaticAnalysisOptions): Promise<StaticAnalysis>;
