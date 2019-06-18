import React, { Component } from 'react'
import Heading from '../Heading/Heading'
import { Container, Collection, CollectionItem, Range, Row, Col, Textarea} from 'react-materialize'

export class Playground extends Component {
  render() {
    return (
      <div>
        <Heading active={true}>Playground</Heading>
        <Container>
          <h6>This page here, is meant to help <i>play</i> with various algorithms, random datasets, etc. 
          and benchmark their performance on your computer (CPU vs GPU) respective to the datasize.</h6>
          <br/>
          <b>SECURITY WARNING</b>: This playground here does an <i>evil eval</i> on the input functions. So, do not carelessly copy and paste things here.
          <br/><br/>
          <div>
            <h4>Step 1) Setup your input parameters generator</h4>
              Setup your various argument generator settings here:<br/>

              <Collection>
                <CollectionItem><b>Number of parameters</b>: To pass to the final GPU.js function</CollectionItem>
                <CollectionItem><b>Sample size</b>: Size for the sample display</CollectionItem>
                <CollectionItem><b>Sample size</b>: Seed value(pseudo random), used by the random function</CollectionItem>
              </Collection>

              Each parameter function is passed in 2 attributes:<br/>

              <Collection>
                <CollectionItem><b>size</b>: The sample size used for this iteration</CollectionItem>
                <CollectionItem><b>rand</b>: The 0-1 floating value, pseudo random generator used for this sample size, and parameter count</CollectionItem>
              </Collection>
              Additionally a dimension generator function, is required to declare the dimensions used<br/>
              <br/>
          </div>
          <div>
            <Row>
              <Col s={6}>
                <div>
                  <label htmlFor="arg_count">Number of Parameters</label>  
                  <div className="input-field">
                    <Range name="arg_count" id="arg_count" defaultValue="3" min="1" max="10" />
                  </div>
                </div>
              </Col>
              <Col s={6}>
                <label htmlFor="sample_size">Sample Size</label>  
                <div className="input-field">
                  <Range name="sample_size" id="sample_size" defaultValue="10" min="1" max="1000" />
                </div>
              </Col>
            </Row>
            <Row>
              <div className="input-field">
                <label htmlFor="rand_seed">Random Seed</label>
                <input type="text" name="rand_seed" id="rand_seed" />
              </div>
            </Row>
          </div>
          <div className="paramContainer">
            <Row>
              <Col s={12} className="dim_group sampleSet" id="dim_group">
                <div className="sampleSetInner input-field">
                  <label htmlFor="dim_function">Dimension Function</label>  
                  <textarea name="dim_function" className="dim_function materialize-textarea" id="dim_function" />
                  <label className="control-label" for="dim_sample">Sample Dimensions</label>  
                  <pre className="dim_sample sample_output" id="dim_sample" />
                </div>
              </Col>
            </Row>
            <button id="paramset_btn" type="button" className="btn btn-primary btn-lg btn-block paramset_btn">Generate parameter samples</button>
          </div>
          <div>
            <h3>Step 2) Program your kernel function</h3>
            <blockquote>
              Code out the kernel, do note the following tips.
              <br/>
              <br/>
              - Parameter functions is automatically called to provide args<br/>
              - if/else is expensive in GPU, if/else in loops is even more expensive
            </blockquote>
          </div>
          <div className=" kernelContainer">
            <div className="kernelGroupInner form-group">
              <label className="control-label" for="param_function">Kernel Function</label><br/>
              <textarea name="kernel_function" className="kernel_function" id="kernel_function"></textarea>
              <button id="kernel_sample_btn" type="button" className="btn btn-primary btn-lg btn-block kernel_sample_btn">Generate kernel sample</button>
              <label className="control-label" for="kernel_sample">Sample Output</label>
              <pre id="kernel_sample" name="kernel_sample" className="kernel_sample"></pre>
            </div>
          </div>
          <div>
            <h3>Step 3) BENCH! CPU vs GPU</h3>
            <blockquote>
              {`Setup your sample size upper, lower bounds. Its increment size. Benchmark iterations. And bench it!<br/><br/>
              Generally speaking however, these are common learning notes.<br/><br/>
              - Due to the non-negligible overhead of running the webgl engine, small data sample sizes (such as <= 250) tends to be slower on GPU. Cut off point varies between kernel and machines`}<br/>
              <br/>
              - There is a small data transfer cost, to move from JS to GPU, paid by the CPU. Which is proportional to the data size. As such extremely simple kernel (such as A+B) will always be slower in GPU<br/>
            </blockquote>
          </div>
          <div>
            <div className="col-sm-4">
              <label className="control-label" for="bench_lower">Lower bounds</label>  
              <div className="input-group">
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="minus" data-field="bench_lower">
                    <span className="glyphicon glyphicon-minus"></span>
                  </button>
                </span>
                <input type="text" name="bench_lower" id="bench_lower" className="form-control input-number" value="50" min="0" max="4294967295" />
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="plus" data-field="bench_lower">
                    <span className="glyphicon glyphicon-plus"></span>
                  </button>
                </span>
              </div>
            </div>
            <div className="col-sm-4">
              <label className="control-label" for="bench_upper">Upper bounds</label>  
              <div className="input-group">
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="minus" data-field="bench_upper">
                    <span className="glyphicon glyphicon-minus"></span>
                  </button>
                </span>
                <input type="text" name="bench_upper" id="bench_upper" className="form-control input-number" value="400" min="0" max="4294967295" />
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="plus" data-field="bench_upper">
                    <span className="glyphicon glyphicon-plus"></span>
                  </button>
                </span>
              </div>
            </div>
            <div className="col-sm-4">
              <label className="control-label" for="bench_increment">Increment size</label>  
              <div className="input-group">
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="minus" data-field="bench_increment">
                    <span className="glyphicon glyphicon-minus"></span>
                  </button>
                </span>
                <input type="text" name="bench_increment" id="bench_increment" className="form-control input-number" value="50" min="0" max="4294967295" />
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="plus" data-field="bench_increment">
                    <span className="glyphicon glyphicon-plus"></span>
                  </button>
                </span>
              </div>
            </div>
            <div className="col-sm-4">
              <label className="control-label" for="arg_count">Bench Size</label>  
              <div className="input-group">
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="minus" data-field="bench_size">
                    <span className="glyphicon glyphicon-minus"></span>
                  </button>
                </span>
                <input type="text" name="bench_size" id="bench_size" className="form-control input-number" value="25" min="0" max="4294967295" />
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="plus" data-field="bench_size">
                    <span className="glyphicon glyphicon-plus"></span>
                  </button>
                </span>
              </div>
            </div>
            <div className="col-sm-4">
              <label className="control-label" for="arg_count">Warmup Size</label>  
              <div className="input-group">
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="minus" data-field="warmup_size">
                    <span className="glyphicon glyphicon-minus"></span>
                  </button>
                </span>
                <input type="text" name="bench_size" id="warmup_size" className="form-control input-number" value="5" min="0" max="4294967295" />
                <span className="input-group-btn">
                  <button type="button" className="btn btn-default btn-number" data-type="plus" data-field="warmup_size">
                    <span className="glyphicon glyphicon-plus"></span>
                  </button>
                </span>
              </div>
            </div>
          </div>
          <div className="chartContainer">
            <button id="bench_btn" type="button" className="btn btn-primary btn-lg btn-block bench_btn">Run the benchmark!</button>
            <h4>Average Time taken</h4>
            <div id="chart_time" className="chart_time bench_chart"></div>
            <h4>GPU Performance improvement</h4>
            <div id="chart_gain" className="chart_gain bench_chart"></div>
          </div>
          <br/>
        </Container>
      </div>
    )
  }
}

export default Playground
