[![Logo](https://gpu.rocks/favicon.png)](https://gpu.rocks/)

[Link to website](https://gpu.rocks/)

### Contributing
**All contributions are welcome**

#### Project Structure:
- src
  - Components: All react components
    - Benchmark (`/#/benchmark`): Definitely needs improvement
    - Content (`/#/`)
    - Example: Example code on the landing page
    - Examples (`/#/examples`)
    - Header: The page header
    - Install (`/#/install`): Installation instructions, can be removed and be linked to the README
    - JellyOnFayyah: Glictchy Jellyfish GIF on landing page
    - Main (`/#/`): Header + Content
    - Nav: Navbar
    - OldVersions: Info about the first release etc., displayed at the bottom of the landing page
    - PageFooter: Footer
    - ScrollButton: A scroll-to-top btn
    - ServerBenchmarks: Graphs with gpu.js benchmarks on a server, displayed on the landing page
    - Strength: Features of gpu.js (like node.js compatibility), need to add more features (like expoGL support) too
    - Syntax: Supported Syntax in gpu.js, displayed on the landing page
    - Util: Common components
      - Code: Syntax Highlighted Codeblock
      - Graph: Graphs
      - Materialicon: easy to use single-component materialIcon
  - Data: All the graphs data
  - db: May want to move the graph data etc. to firebase in the future
  - img: Images and GIFs
  - scss: Global SASS files apart from the ones per component.

#### Development
The site is a [Vite](https://vitejs.dev/) + React app. Requires a current Node (18+).

```sh
yarn install
yarn dev      # dev server on http://localhost:3000
yarn build    # production build into dist/
yarn preview  # serve the production build locally
yarn deploy   # build and publish dist/ to the gh-pages branch
```

Two checks run against a real browser (they use `puppeteer-core`, which does
not download anything — set `CHROME_PATH` if Chrome is somewhere unusual):

```sh
yarn test:smoke                                  # every route rendered, /api/ still served
yarn test:compare https://gpu.rocks http://localhost:4173   # diff two deployments
```

`test:smoke` runs in CI on every push and pull request. It exists because a
build can succeed and still ship a blank site — the move off create-react-app
did exactly that until a missing `global` shim was found. `test:compare` is a
manual tool for changes that should *not* alter the site, such as a build tool
swap or a dependency bump; it diffs the rendered text, links and images of two
deployments.

Everything in `public/` is copied verbatim into the build — that includes
`api/` (the generated gpu.js API reference served at https://gpu.rocks/api/),
`404.html` (forwards deep links to the hash routes), and `CNAME`.

Note `public/service-worker.js` is deliberately a no-op that unregisters
itself: the site used to ship a create-react-app service worker whose
navigation fallback served the app shell for every extension-less path, so
`/api/` rendered the homepage (gpujs/gpu.js#852). Keep the file so browsers
still holding the old worker tear it down.

#### Currently Followed Conventions:
##### Coding Conventions
- Single line code is ended with a `;` but multiline without it.

##### Naming
- All components are put inside separate directories with the same name as the component
- The main component file starts with a capital letter, is camelcased, and uses the `.jsx` extension (files containing JSX must, so Vite can transform them).
- Each component directory may have a `scss` file specific to that component, and should have the same name as the component.
- The component directories can have anything else.