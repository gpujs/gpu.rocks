"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const path_1 = require("path");
const stream_utils_1 = require("../../stream-utils");
const nodeAppFiles = ["package.json", "package-lock.json", "yarn.lock"];
function filePathMatches(filePath) {
    const fileName = path_1.basename(filePath);
    return (filePath.indexOf("node_modules") === -1 && nodeAppFiles.includes(fileName));
}
exports.getNodeAppFileContentAction = {
    actionName: "node-app-files",
    filePathMatches,
    callback: stream_utils_1.streamToString,
};
function getNodeAppFileContent(extractedLayers) {
    const foundAppFiles = {};
    for (const filePath of Object.keys(extractedLayers)) {
        for (const actionName of Object.keys(extractedLayers[filePath])) {
            if (actionName !== exports.getNodeAppFileContentAction.actionName) {
                continue;
            }
            if (!(typeof extractedLayers[filePath][actionName] === "string")) {
                throw new Error("expected string");
            }
            foundAppFiles[filePath] = extractedLayers[filePath][actionName];
        }
    }
    return foundAppFiles;
}
exports.getNodeAppFileContent = getNodeAppFileContent;
//# sourceMappingURL=static.js.map