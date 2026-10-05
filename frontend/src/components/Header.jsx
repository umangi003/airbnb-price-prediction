export default function Header() {
  return (
    <header className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 py-4 md:py-6 flex items-center justify-between">
        {/* Logo & Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-cherry-red to-red-600 rounded-lg flex items-center justify-center text-white font-bold text-lg shadow-lg">
            A
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-bold text-white">
              Airbnb Pricing
            </h1>
            <p className="text-xs md:text-sm text-gray-400">
              AI-Powered Price Optimization
            </p>
          </div>
        </div>

        {/* Right Side Info */}
        <div className="hidden md:flex items-center gap-6">
          <div className="text-right">
            <p className="text-sm text-gray-400">Powered by</p>
            <p className="text-white font-semibold">4 ML Models</p>
          </div>
        </div>
      </div>
    </header>
  );
}