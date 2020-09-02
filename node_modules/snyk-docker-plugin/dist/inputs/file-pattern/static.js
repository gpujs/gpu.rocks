"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const minimatch = require("minimatch");
const path = require("path");
const stream_utils_1 = require("../../stream-utils");
function generatePathMatcher(globsInclude, globsExclude) {
    return (filePath) => {
        let exclude = false;
        for (const g of globsExclude) {
            if (!exclude && minimatch(filePath, g)) {
                exclude = true;
            }
        }
        if (!exclude) {
            for (const g of globsInclude) {
                if (minimatch(filePath, g)) {
                    return true;
                }
            }
        }
        return false;
    };
}
function generateExtractAction(globsInclude, globsExclude) {
    return {
        actionName: "find-files-by-pattern",
        filePathMatches: generatePathMatcher(globsInclude, globsExclude),
        callback: stream_utils_1.streamToBuffer,
    };
}
exports.generateExtractAction = generateExtractAction;
function getMatchingFiles(extractedLayers) {
    const manifestFiles = [];
    for (const filePath of Object.keys(extractedLayers)) {
        for (const actionName of Object.keys(extractedLayers[filePath])) {
            if (actionName !== "find-files-by-pattern") {
                continue;
            }
            if (!Buffer.isBuffer(extractedLayers[filePath][actionName])) {
                throw new Error("expected a buffer");
            }
            manifestFiles.push({
                name: path.basename(filePath),
                path: path.dirname(filePath),
                contents: extractedLayers[filePath][actionName],
            });
        }
    }
    return manifestFiles;
}
exports.getMatchingFiles = getMatchingFiles;
//# sourceMappingURL=static.js.map