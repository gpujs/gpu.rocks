import { ExtractAction, ExtractedLayers } from "../../extractor/types";
export declare const getNodeAppFileContentAction: ExtractAction;
export declare function getNodeAppFileContent(extractedLayers: ExtractedLayers): {
    [fileName: string]: string;
};
