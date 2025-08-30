import './App.css';
import { useState } from '@lynx-js/react';
import HomePage from './components/Home.js';
import VideosPage from './components/Videos.js';
import NavBar from './components/NavBar.js';

type AppProps = {
  onRender?: () => void;
};

export function App({ onRender }: AppProps) {
  
  onRender?.();

  const [currentPage, setCurrentPage] = useState('home');

  return (
    <view>
      { currentPage === 'home' && <HomePage />}
      { currentPage === 'videos' && <VideosPage />}
      <NavBar setCurrentPage={setCurrentPage} />
    </view>
  );
}