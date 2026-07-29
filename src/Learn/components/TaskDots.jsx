import React from 'react';

// Mini kernel-grid progress dots (see .taskdots in scss/learn.scss).
// props: total (dot count), doneCount (first N shown done),
// currentIndex (0-based dot shown as 'now'; pass -1 for none).
function TaskDots({ total, doneCount = 0, currentIndex = -1 }) {
  const dots = [];
  for (let i = 0; i < total; i++) {
    let cls;
    if (i === currentIndex) cls = 'now';
    else if (i < doneCount) cls = 'done';
    dots.push(<i key={i} className={cls} />);
  }
  return <div className="taskdots">{dots}</div>;
}

export default TaskDots;
