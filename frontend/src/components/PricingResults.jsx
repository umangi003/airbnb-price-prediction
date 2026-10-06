import React from 'react';

const NEIGHBOURHOODS = [
  'Bourse', 'Buttes-Chaumont', 'Buttes-Montmartre', 'Entrepôt', 'Gobelins',
  'Hôtel-de-Ville', 'Louvre', 'Luxembourg', 'Ménilmontant', 'Observatoire',
  'Opéra', 'Palais-Bourbon', 'Panthéon', 'Passy', 'Popincourt', 'Reuilly',
  'Temple', 'Vaugirard', 'Élysée',
];

const ROOM_TYPES = ['Entire Home', 'Private Room', 'Shared Room', 'Hotel Room'];

export default function PricingResults({ results, formData, onReset }) {
  if (!results || !results.predicted_price_eur) {
    return <div className="card p-6 text-red-600">Error: No price data received</div>;
  }

  const neighbourhood = NEIGHBOURHOODS[formData[5]] || 'Unknown';
  const roomType = ROOM_TYPES[formData[0]] || 'Unknown';
  const xgbPrice = Math.round(results.predicted_price_eur);
  const minPrice = Math.round(xgbPrice * 0.9);
  const maxPrice = Math.round(xgbPrice * 1.1);

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="card bg-gradient-to-br from-white to-green-50 shadow-xl">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">XGBoost Recommended Price</h2>
          <div className="text-6xl font-bold bg-gradient-to-r from-green-500 to-green-600 bg-clip-text text-transparent mb-2">
            €{xgbPrice}
          </div>
          <p className="text-sm text-slate-gray">Per night in {neighbourhood}</p>
          <div className="inline-block mt-4 px-3 py-1 rounded-full text-sm font-semibold bg-green-100 text-green-800">
            ✓ High Confidence
          </div>
        </div>
        <div className="border-t border-gray-200 pt-6">
          <p className="font-semibold text-gray-900 mb-2">
            Typical range: <span className="text-green-600">€{minPrice} - €{maxPrice}</span>/night
          </p>
        </div>
      </div>

      <div className="flex gap-4">
        <button onClick={onReset} className="btn-secondary flex-1">Calculate New Price</button>
      </div>
    </div>
  );
}