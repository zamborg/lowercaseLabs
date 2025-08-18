import React from 'react';
import {
    BrowserRouter as Router,
    Routes,
    Route,
    Link
} from 'react-router-dom';
import Transactions from './components/Transactions';
import Sankey from './components/Sankey';
import TinderUI from './components/TinderUI';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav>
          <ul>
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/transactions">Transactions</Link>
            </li>
            <li>
              <Link to="/sankey">Sankey</Link>
            </li>
            <li>
              <Link to="/tinder">Tinder</Link>
            </li>
          </ul>
        </nav>

        <Routes>
          <Route path="/transactions" element={<Transactions />} />
          <Route path="/sankey" element={<Sankey />} />
          <Route path="/tinder" element={<TinderUI />} />
          <Route path="/" element={<Home />} />
        </Routes>
      </div>
    </Router>
  );
}

const Home = () => <h1>Welcome to Zudget</h1>;

export default App;
