import { DockerFileAnalysis } from "../../docker-file";
import { ExtractedLayers } from "../../extractor/types";
import { OSRelease } from "../types";
export declare function detect(extractedLayers: ExtractedLayers, dockerfileAnalysis: DockerFileAnalysis | undefined): Promise<OSRelease>;
