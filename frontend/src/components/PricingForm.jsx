import React, { useState } from 'react';

const NEIGHBOURHOODS = [
  'Bourse',
  'Buttes-Chaumont',
  'Buttes-Montmartre',
  'Entrepôt',
  'Gobelins',
  'Hôtel-de-Ville',
  'Louvre',
  'Luxembourg',
  'Ménilmontant',
  'Observatoire',
  'Opéra',
  'Palais-Bourbon',
  'Panthéon',
  'Passy',
  'Popincourt',
  'Reuilly',
  'Temple',
  'Vaugirard',
  'Élysée',
];

const ROOM_TYPES = ['Entire Home', 'Private Room', 'Shared Room', 'Hotel Room'];

const ROOM_TYPE_CONSTRAINTS = {
  'Entire Home': { max_bedrooms: 6, max_bathrooms: 4, max_occupancy: 10 },
  'Private Room': { max_bedrooms: 2, max_bathrooms: 2, max_occupancy: 4 },
  'Shared Room': { max_bedrooms: 1, max_bathrooms: 1, max_occupancy: 2 },
  'Hotel Room': { max_bedrooms: 1, max_bathrooms: 1, max_occupancy: 3 },
};

const FIELD_ICONS = {
  room_type: '🏠',
  bedrooms: '🛏️',
  bathrooms: '🛁',
  accommodates: '👥',
  minimum_stay: '📅',
  neighbourhood: '📍',
  air_conditioning: '❄️',
  availability_90: '📊',
  is_superhost: '⭐',
  review_rating: '⭐',
};

export default function PricingForm({ onPrediction, isLoading }) {
  const [formData, setFormData] = useState({
    room_type: '',
    bedrooms: '',
    bathrooms: '',
    accommodates: '',
    minimum_stay: '',
    neighbourhood: '',
    air_conditioning: false,
    availability_90: 80,
    is_superhost: false,
    review_rating: 4.5,
  });

  const [errors, setErrors] = useState({});

  const getConstraints = () => {
    return ROOM_TYPE_CONSTRAINTS[formData.room_type] || { max_bedrooms: 6, max_bathrooms: 4, max_occupancy: 10 };
  };

  const validateForm = () => {
    const newErrors = {};
    const constraints = getConstraints();

    if (!formData.room_type) newErrors.room_type = 'Room type is required';
    if (!formData.bedrooms || formData.bedrooms < 1) {
      newErrors.bedrooms = 'Bedrooms is required';
    } else if (formData.bedrooms > constraints.max_bedrooms) {
      newErrors.bedrooms = `Maximum bedrooms for ${formData.room_type} is ${constraints.max_bedrooms}`;
    }
    if (!formData.bathrooms || formData.bathrooms < 1) {
      newErrors.bathrooms = 'Bathrooms is required';
    } else if (formData.bathrooms > constraints.max_bathrooms) {
      newErrors.bathrooms = `Maximum bathrooms for ${formData.room_type} is ${constraints.max_bathrooms}`;
    }
    if (!formData.accommodates || formData.accommodates < 1) {
      newErrors.accommodates = 'Accommodates is required';
    } else if (formData.accommodates > constraints.max_occupancy) {
      newErrors.accommodates = `Maximum occupancy for ${formData.room_type} is ${constraints.max_occupancy} guests`;
    }
    if (!formData.minimum_stay || formData.minimum_stay < 1 || formData.minimum_stay > 90) {
      newErrors.minimum_stay = 'Minimum stay must be between 1 and 90 nights';
    }
    if (!formData.neighbourhood) newErrors.neighbourhood = 'Neighbourhood is required';

    return newErrors;
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;

    setFormData(prev => ({
      ...prev,
      [name]: newValue,
    }));

    if (errors[name]) {
      const newErrors = { ...errors };
      delete newErrors[name];
      setErrors(newErrors);
    }
  };

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    if (name === 'review_rating') {
      setFormData(prev => ({
        ...prev,
        review_rating: parseFloat(value),
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        availability_90: parseInt(value, 10),
      }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const newErrors = validateForm();
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    const predictionData = [
      ROOM_TYPES.indexOf(formData.room_type),
      parseFloat(formData.bedrooms),
      parseFloat(formData.bathrooms),
      parseFloat(formData.accommodates),
      parseFloat(formData.minimum_stay),
      NEIGHBOURHOODS.indexOf(formData.neighbourhood),
      formData.air_conditioning ? 1 : 0,
      formData.availability_90 / 100,
      formData.is_superhost ? 1 : 0,
      formData.review_rating,
    ];

    onPrediction(predictionData);
  };

  const isFormValid =
    formData.room_type &&
    formData.bedrooms &&
    formData.bathrooms &&
    formData.accommodates &&
    formData.minimum_stay &&
    formData.neighbourhood;

  const constraints = getConstraints();

  return (
    <>
      {!isLoading ? (
        <form onSubmit={handleSubmit} className="card shadow-xl">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-1">About Your Listing</h2>
            <p className="text-slate-gray">Tell us about your property</p>
          </div>

          <div className="space-y-6">
            {/* Room Type */}
            <div className="animate-slideUp" style={{ animationDelay: '0.1s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.room_type}</span>
                Room Type <span className="text-red-600">*</span>
              </label>
              <select
                name="room_type"
                value={formData.room_type}
                onChange={handleChange}
                className="select-field focus:ring-2 focus:ring-cherry-red focus:border-transparent"
              >
                <option value="">Select room type...</option>
                {ROOM_TYPES.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
              {errors.room_type && <p className="text-error">{errors.room_type}</p>}
              <p className="text-help">Select the type of accommodation</p>
            </div>

            {/* Bedrooms */}
            <div className="animate-slideUp" style={{ animationDelay: '0.2s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.bedrooms}</span>
                Bedrooms <span className="text-red-600">*</span>
              </label>
              <input
                type="number"
                name="bedrooms"
                value={formData.bedrooms}
                onChange={handleChange}
                min="1"
                max={constraints.max_bedrooms}
                placeholder="1"
                className="input-field focus:ring-2 focus:ring-cherry-red focus:border-transparent"
                disabled={!formData.room_type}
              />
              {errors.bedrooms && <p className="text-error">{errors.bedrooms}</p>}
              <p className="text-help">
                {formData.room_type 
                  ? `Maximum for ${formData.room_type}: ${constraints.max_bedrooms}`
                  : 'Select room type first'}
              </p>
            </div>

            {/* Bathrooms */}
            <div className="animate-slideUp" style={{ animationDelay: '0.3s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.bathrooms}</span>
                Bathrooms <span className="text-red-600">*</span>
              </label>
              <input
                type="number"
                name="bathrooms"
                value={formData.bathrooms}
                onChange={handleChange}
                min="1"
                max={constraints.max_bathrooms}
                step="0.5"
                placeholder="1"
                className="input-field focus:ring-2 focus:ring-cherry-red focus:border-transparent"
                disabled={!formData.room_type}
              />
              {errors.bathrooms && <p className="text-error">{errors.bathrooms}</p>}
              <p className="text-help">
                {formData.room_type 
                  ? `Maximum for ${formData.room_type}: ${constraints.max_bathrooms}`
                  : 'Select room type first'}
              </p>
            </div>

            {/* Accommodates */}
            <div className="animate-slideUp" style={{ animationDelay: '0.4s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.accommodates}</span>
                Maximum Guests <span className="text-red-600">*</span>
              </label>
              <input
                type="number"
                name="accommodates"
                value={formData.accommodates}
                onChange={handleChange}
                min="1"
                max={constraints.max_occupancy}
                placeholder="2"
                className="input-field focus:ring-2 focus:ring-cherry-red focus:border-transparent"
                disabled={!formData.room_type}
              />
              {errors.accommodates && <p className="text-error">{errors.accommodates}</p>}
              <p className="text-help">
                {formData.room_type 
                  ? `Maximum for ${formData.room_type}: ${constraints.max_occupancy} guests`
                  : 'Select room type first'}
              </p>
            </div>

            {/* Minimum Stay */}
            <div className="animate-slideUp" style={{ animationDelay: '0.5s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.minimum_stay}</span>
                Minimum Stay <span className="text-red-600">*</span>
              </label>
              <input
                type="number"
                name="minimum_stay"
                value={formData.minimum_stay}
                onChange={handleChange}
                min="1"
                max="90"
                placeholder="1"
                className="input-field focus:ring-2 focus:ring-cherry-red focus:border-transparent"
              />
              {errors.minimum_stay && <p className="text-error">{errors.minimum_stay}</p>}
              <p className="text-help">Required minimum nights for booking</p>
            </div>

            {/* Neighbourhood */}
            <div className="animate-slideUp" style={{ animationDelay: '0.6s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.neighbourhood}</span>
                Neighbourhood <span className="text-red-600">*</span>
              </label>
              <select
                name="neighbourhood"
                value={formData.neighbourhood}
                onChange={handleChange}
                className="select-field focus:ring-2 focus:ring-cherry-red focus:border-transparent"
              >
                <option value="">Select your neighbourhood...</option>
                {NEIGHBOURHOODS.map(neighbourhood => (
                  <option key={neighbourhood} value={neighbourhood}>{neighbourhood}</option>
                ))}
              </select>
              {errors.neighbourhood && <p className="text-error">{errors.neighbourhood}</p>}
              <p className="text-help">The location significantly affects pricing</p>
            </div>

            {/* Air Conditioning */}
            <div className="animate-slideUp" style={{ animationDelay: '0.7s' }}>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  name="air_conditioning"
                  checked={formData.air_conditioning}
                  onChange={handleChange}
                  className="toggle-checkbox"
                />
                <span className="font-bold text-gray-900 flex items-center gap-2">
                  <span>{FIELD_ICONS.air_conditioning}</span>
                  Does your listing have air conditioning?
                </span>
              </label>
              <p className="text-help">Premium amenity that increases price</p>
            </div>

            {/* 90-Day Availability */}
            <div className="animate-slideUp" style={{ animationDelay: '0.8s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.availability_90}</span>
                90-Day Availability
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  name="availability_90"
                  value={formData.availability_90}
                  onChange={handleSliderChange}
                  min="0"
                  max="100"
                  className="flex-1 h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-cherry-red"
                />
                <span className="font-bold text-cherry-red text-lg min-w-12">
                  {formData.availability_90}%
                </span>
              </div>
              <p className="text-help">What percentage of next 90 days is available for booking?</p>
            </div>

            {/* Superhost Status */}
            <div className="animate-slideUp" style={{ animationDelay: '0.9s' }}>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  name="is_superhost"
                  checked={formData.is_superhost}
                  onChange={handleChange}
                  className="toggle-checkbox"
                />
                <span className="font-bold text-gray-900 flex items-center gap-2">
                  <span>{FIELD_ICONS.is_superhost}</span>
                  Are you a Superhost?
                </span>
              </label>
              <p className="text-help">Superhost status can increase your booking rate</p>
            </div>

            {/* Review Rating */}
            <div className="animate-slideUp" style={{ animationDelay: '1s' }}>
              <label className="block font-bold text-gray-900 mb-2 flex items-center gap-2">
                <span>{FIELD_ICONS.review_rating}</span>
                Your Average Review Score
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  name="review_rating"
                  value={formData.review_rating}
                  onChange={handleSliderChange}
                  min="0"
                  max="5"
                  step="0.1"
                  className="flex-1 h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-cherry-red"
                />
                <span className="font-bold text-cherry-red text-lg min-w-12">
                  {formData.review_rating.toFixed(1)}
                </span>
              </div>
              <p className="text-help">Based on guest reviews (0-5 stars)</p>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={!isFormValid || isLoading}
            className="btn-primary mt-8 w-full shadow-lg hover:shadow-xl transition-shadow"
          >
            {isLoading ? 'Analyzing...' : 'See My Recommended Price'}
          </button>
        </form>
      ) : (
        <div className="card w-full shadow-xl">
          <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-gray-300 border-t-cherry-red rounded-full animate-spin"></div>
            <p className="text-center text-slate-gray">
              Analyzing similar listings in your area...
            </p>
          </div>
        </div>
      )}
    </>
  );
}