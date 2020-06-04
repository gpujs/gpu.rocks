"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.REACHABLE_VULNS_SUPPORTED_PACKAGE_MANAGERS = exports.PINNING_SUPPORTED_PACKAGE_MANAGERS = exports.GRAPH_SUPPORTED_PACKAGE_MANAGERS = exports.PROTECT_SUPPORTED_PACKAGE_MANAGERS = exports.WIZARD_SUPPORTED_PACKAGE_MANAGERS = exports.SUPPORTED_PACKAGE_MANAGER_NAME = void 0;
exports.SUPPORTED_PACKAGE_MANAGER_NAME = {
    rubygems: 'RubyGems',
    npm: 'npm',
    yarn: 'Yarn',
    maven: 'Maven',
    pip: 'pip',
    sbt: 'SBT',
    gradle: 'Gradle',
    golangdep: 'dep (Go)',
    gomodules: 'Go Modules',
    govendor: 'govendor',
    nuget: 'NuGet',
    paket: 'Paket',
    composer: 'Composer',
    cocoapods: 'CocoaPods',
};
exports.WIZARD_SUPPORTED_PACKAGE_MANAGERS = [
    'yarn',
    'npm',
];
exports.PROTECT_SUPPORTED_PACKAGE_MANAGERS = [
    'yarn',
    'npm',
];
exports.GRAPH_SUPPORTED_PACKAGE_MANAGERS = [
    'npm',
    'sbt',
    'yarn',
    'rubygems',
];
// For ecosystems with a flat set of libraries (e.g. Python, JVM), one can
// "pin" a transitive dependency
exports.PINNING_SUPPORTED_PACKAGE_MANAGERS = [
    'pip',
];
exports.REACHABLE_VULNS_SUPPORTED_PACKAGE_MANAGERS = [
    'maven',
];
//# sourceMappingURL=package-managers.js.map