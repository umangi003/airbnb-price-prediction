import { useState } from 'react';
import './index.css';
import PricingForm from './components/PricingForm';
import PricingResults from './components/PricingResults';
import Header from './components/Header';
import Footer from './components/Footer';

function App() {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState(null);

  const handlePrediction = (data) => {
    setFormData(data);
    setIsLoading(true);

    fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data: data }),
    })
      .then(response => {
        if (!response.ok) throw new Error('Backend connection failed');
        return response.json();
      })
      .then(result => {
        setResults(result);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Prediction error:', error);
        setIsLoading(false);
        alert('Something went wrong. Please try again.');
      });
  };

  const handleReset = () => {
    setResults(null);
    setFormData(null);
  };

  return (
    <div className="min-h-screen w-full bg-eggshell flex flex-col">
      <Header />

      {!results && (
        <div className="bg-gradient-to-r from-cherry-red via-red-600 to-red-700 py-16 md:py-20 text-white">
          <div className="max-w-6xl mx-auto px-4 text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              Know Your Perfect Price
            </h1>
            <p className="text-lg md:text-xl text-red-100">
              AI-powered pricing recommendations for Paris Airbnb hosts in seconds
            </p>
          </div>
        </div>
      )}

      <main className="w-full flex-1">
        <div className="max-w-7xl mx-auto px-4 py-12 md:py-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
            {/* Left Column - Form */}
            <div className="animate-fadeIn">
              <PricingForm onPrediction={handlePrediction} isLoading={isLoading} />
            </div>

            {/* Right Column - Results */}
            {(results || isLoading) && (
              <div id="results-section" className="animate-fadeIn">
                {isLoading ? (
                  <div className="card w-full shadow-xl">
                    <div className="flex flex-col items-center gap-4">
                      <div className="w-12 h-12 border-4 border-gray-300 border-t-cherry-red rounded-full animate-spin"></div>
                      <p className="text-center text-slate-gray">
                        Analyzing similar listings in your area...
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    <PricingResults
                      results={results}
                      formData={formData}
                      onReset={handleReset}
                    />
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;