import { ScannedProjectCustom } from "../../types";
import { FilePathToContent } from "./types";
export declare function nodeFilesToScannedProjects(filePathToContent: FilePathToContent): Promise<ScannedProjectCustom[]>;
