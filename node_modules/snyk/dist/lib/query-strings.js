"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getQueryParamsAsString = void 0;
const url = require("url");
const os = require("os");
const is_docker_1 = require("./is-docker");
function getQueryParamsAsString() {
    var _a;
    const SNYK_UTM_MEDIUM = process.env.SNYK_UTM_MEDIUM || 'cli';
    const SNYK_UTM_SOURCE = process.env.SNYK_UTM_SOURCE || 'cli';
    const SNYK_UTM_CAMPAIGN = process.env.SNYK_UTM_CAMPAIGN || 'cli';
    const osType = (_a = os.type()) === null || _a === void 0 ? void 0 : _a.toLowerCase();
    const docker = is_docker_1.isDocker().toString();
    /* eslint-disable @typescript-eslint/camelcase */
    const queryParams = new url.URLSearchParams({
        utm_medium: SNYK_UTM_MEDIUM,
        utm_source: SNYK_UTM_SOURCE,
        utm_campaign: SNYK_UTM_CAMPAIGN,
        os: osType,
        docker,
    });
    /* eslint-enable @typescript-eslint/camelcase */
    return queryParams.toString();
}
exports.getQueryParamsAsString = getQueryParamsAsString;
//# sourceMappingURL=query-strings.js.map