import React from 'react';
import { Routes, Route, Navigate, useParams } from 'react-router';
import { ThemeProvider, useTheme } from './ThemeContext';
import LearnHome from './home/LearnHome';
import TaskPage from './task/TaskPage';
import VerifyPage from './engine/VerifyPage';
import { getModule } from './content/index';
import { moduleProgress } from './engine/storage';
import './scss/learn.scss';

// /learn/:moduleId → first incomplete task (task 1 when the module is
// unknown, untouched, or fully complete).
function ModuleRedirect() {
  const { moduleId } = useParams();
  const module = getModule(moduleId);
  if (!module) return <Navigate to="/learn" replace />;
  const { currentIndex } = moduleProgress(module);
  const taskNum = currentIndex === -1 ? 1 : currentIndex + 1;
  return <Navigate to={`/learn/${moduleId}/${taskNum}`} replace />;
}

// Everything inside .learn-root is scoped away from the site's global CSS.
// data-theme always carries the *effective* theme ('light' | 'dark');
// 'auto' is resolved in ThemeContext via matchMedia.
function LearnRoot({ children }) {
  const { theme } = useTheme();
  return (
    <div className="learn-root" data-theme={theme}>
      <div className="screen">{children}</div>
    </div>
  );
}

// Mounted at /learn/* (course UI) and at /learn-verify (verify prop set —
// the headless test hook used by scripts/verify-learn.mjs).
function LearnApp({ verify = false }) {
  return (
    <ThemeProvider>
      <LearnRoot>
        {verify ? (
          <VerifyPage />
        ) : (
          <Routes>
            <Route index element={<LearnHome />} />
            <Route path=":moduleId" element={<ModuleRedirect />} />
            <Route path=":moduleId/:taskNum" element={<TaskPage />} />
            <Route path="*" element={<Navigate to="/learn" replace />} />
          </Routes>
        )}
      </LearnRoot>
    </ThemeProvider>
  );
}

export default LearnApp;
