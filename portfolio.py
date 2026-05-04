import numpy as np

def calculate_portfolio(returns, risk_level="Medium"):
    num_assets = len(returns.columns)
    
    best_sharpe = -999
    best_weights = None
    
    for _ in range(3000):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(returns.mean() * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
        
        sharpe = portfolio_return / portfolio_risk
        
        # Risk filtering
        if risk_level == "Low" and portfolio_risk > 0.02:
            continue
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
    
    return best_weights, best_sharpe