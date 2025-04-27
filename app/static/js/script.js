document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const tickerInput = document.getElementById('ticker-input');
    const timeframeSelect = document.getElementById('timeframe-select');
    const predictBtn = document.getElementById('predict-btn');
    const loadingElement = document.getElementById('loading');
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');
    const resultContainer = document.getElementById('result-container');
    const stockTicker = document.getElementById('stock-ticker');
    const currentPrice = document.getElementById('current-price');
    const lastUpdated = document.getElementById('last-updated');
    const prediction = document.getElementById('prediction');
    const confidence = document.getElementById('confidence');
    const probability = document.getElementById('probability');
    const mainContainer = document.querySelector('main');
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = themeToggle.querySelector('i');
    
    // Chart elements and state
    const chartCanvas = document.getElementById('historical-chart');
    let priceChart = null;
    let chartData = {
        labels: [],
        prices: []
    };

    // Section containers
    const technicalIndicators = document.getElementById('technical-indicators');
    const newsSentiment = document.getElementById('news-sentiment');

    // Theme state
    let isDarkTheme = true;

    // Add animated background
    createStockBackground();

    // Enable input animation
    tickerInput.addEventListener('focus', function() {
        this.parentElement.classList.add('focused');
    });
    
    tickerInput.addEventListener('blur', function() {
        this.parentElement.classList.remove('focused');
    });

    // Event listeners
    predictBtn.addEventListener('click', getPrediction);
    tickerInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getPrediction();
        }
    });
    
    // Theme toggle event listener
    themeToggle.addEventListener('click', toggleTheme);

    // Add button ripple effect
    predictBtn.addEventListener('mousedown', createRipple);

    function createRipple(event) {
        const button = event.currentTarget;
        
        const circle = document.createElement('span');
        const diameter = Math.max(button.clientWidth, button.clientHeight);
        
        circle.style.width = circle.style.height = `${diameter}px`;
        circle.style.left = `${event.clientX - button.getBoundingClientRect().left - diameter / 2}px`;
        circle.style.top = `${event.clientY - button.getBoundingClientRect().top - diameter / 2}px`;
        circle.classList.add('ripple');
        
        const ripple = button.querySelector('.ripple');
        if (ripple) {
            ripple.remove();
        }
        
        button.appendChild(circle);
        
        // Remove ripple after animation completes
        setTimeout(() => {
            circle.remove();
        }, 600);
    }

    // Function to get stock prediction
    function getPrediction() {
        const ticker = tickerInput.value.trim().toUpperCase();
        const days = Number(timeframeSelect.value);
        
        if (!ticker) {
            showError('Please enter a stock symbol');
            return;
        }

        // Show loading spinner with animation
        mainContainer.classList.add('loading-state');
        errorContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
        
        // Animate the loading in
        loadingElement.classList.remove('hidden');
        loadingElement.style.opacity = '0';
        setTimeout(() => {
            loadingElement.style.opacity = '1';
        }, 10);

        // Update the input value with uppercase ticker
        tickerInput.value = ticker;

        // Add loading effect to button
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;

        // Make API request to get prediction
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ticker: ticker, days: days }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to get prediction');
                });
            }
            return response.json();
        })
        .then(data => {
            // Add delay for smoother transition
            setTimeout(() => {
                displayResult(data);
                mainContainer.classList.remove('loading-state');
                predictBtn.classList.remove('loading');
                predictBtn.disabled = false;
            }, 500);
        })
        .catch(error => {
            setTimeout(() => {
                showError(error.message);
                mainContainer.classList.remove('loading-state');
                predictBtn.classList.remove('loading');
                predictBtn.disabled = false;
            }, 500);
        });
    }

    // Function to display prediction result with animation
    function displayResult(data) {
        // Hide loading with fade out
        loadingElement.style.opacity = '0';
        setTimeout(() => {
            loadingElement.classList.add('hidden');
            
            // Update stock info
            stockTicker.textContent = data.ticker;
            currentPrice.textContent = `Current Price: $${data.current_price}`;
            lastUpdated.textContent = `Last Updated: ${data.last_updated}`;
    
            // Update prediction
            prediction.textContent = data.prediction;
            prediction.className = 'prediction-value';
            prediction.classList.add(data.prediction.toLowerCase());
    
            // Update confidence
            confidence.textContent = `${data.confidence} Confidence`;
            confidence.className = 'confidence';
            confidence.classList.add(data.confidence.toLowerCase());
    
            // Update probability
            probability.textContent = `Probability: ${data.probability}%`;
            
            // Create historical chart
            if (data.historical_data && data.historical_data.length > 0) {
                createHistoricalChart(data.historical_data);
            }
    
            // Display technical indicators if available
            if (data.technical_indicators && !data.technical_indicators.error) {
                displayTechnicalIndicators(data.technical_indicators);
                technicalIndicators.classList.remove('hidden');
            } else {
                technicalIndicators.classList.add('hidden');
            }
            
            // NEW: Display news sentiment if available
            if (data.news_sentiment && !data.news_sentiment.error) {
                displayNewsSentiment(data.news_sentiment);
                newsSentiment.classList.remove('hidden');
            } else {
                newsSentiment.classList.add('hidden');
            }
    
            // Show result container with animation
            resultContainer.classList.remove('hidden');
            resultContainer.style.opacity = '0';
            resultContainer.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                resultContainer.style.opacity = '1';
                resultContainer.style.transform = 'translateY(0)';
            }, 50);
            
            // Add confetti animation for high confidence UP predictions
            if (data.prediction === 'UP' && data.confidence === 'High') {
                showConfetti();
            }
        }, 300);
    }

    // Function to create or update historical price chart
    function createHistoricalChart(historicalData) {
        // Prepare chart data
        chartData.labels = historicalData.map(item => item.date);
        chartData.prices = historicalData.map(item => item.price);
        
        // Set chart theme colors based on current theme
        const themeColors = {
            backgroundColor: 'rgba(58, 123, 213, 0.1)',
            borderColor: 'rgba(58, 123, 213, 0.8)',
            pointBackgroundColor: 'rgba(58, 123, 213, 1)',
            gridColor: 'rgba(0, 0, 0, 0.1)',
            textColor: '#000000'  // Always black text for maximum visibility
        };
        
        // Destroy existing chart if it exists
        if (priceChart) {
            priceChart.destroy();
        }
        
        // Create new chart
        priceChart = new Chart(chartCanvas, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'Stock Price ($)',
                    data: chartData.prices,
                    backgroundColor: themeColors.backgroundColor,
                    borderColor: themeColors.borderColor,
                    borderWidth: 2,
                    pointBackgroundColor: themeColors.pointBackgroundColor,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: themeColors.textColor,
                            font: {
                                weight: 'bold'
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: function(context) {
                                return `Price: $${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: themeColors.gridColor
                        },
                        ticks: {
                            color: themeColors.textColor,
                            maxRotation: 45,
                            minRotation: 45,
                            font: {
                                weight: 'bold'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date',
                            color: themeColors.textColor,
                            font: {
                                weight: 'bold',
                                size: 14
                            }
                        }
                    },
                    y: {
                        grid: {
                            color: themeColors.gridColor
                        },
                        ticks: {
                            color: themeColors.textColor,
                            font: {
                                weight: 'bold'
                            },
                            callback: function(value) {
                                return '$' + value;
                            }
                        },
                        title: {
                            display: true,
                            text: 'Price (USD)',
                            color: themeColors.textColor,
                            font: {
                                weight: 'bold',
                                size: 14
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    // Function to show error message with animation
    function showError(message) {
        // Hide loading with fade out
        loadingElement.style.opacity = '0';
        setTimeout(() => {
            loadingElement.classList.add('hidden');
            
            errorText.textContent = message;
            errorContainer.classList.remove('hidden');
            errorContainer.style.opacity = '0';
            errorContainer.style.transform = 'translateX(-10px)';
            
            setTimeout(() => {
                errorContainer.style.opacity = '1';
                errorContainer.style.transform = 'translateX(0)';
            }, 50);
            
            resultContainer.classList.add('hidden');
        }, 300);
    }
    
    // Create animated stock background
    function createStockBackground() {
        const backgroundContainer = document.createElement('div');
        backgroundContainer.classList.add('stock-background');
        
        // Create random stock chart lines
        for (let i = 0; i < 3; i++) {
            const line = document.createElement('div');
            line.classList.add('stock-line');
            line.style.animationDelay = `${i * 0.5}s`;
            line.style.bottom = `${15 + i * 30}%`;
            line.style.opacity = 0.05 + (i * 0.02);
            backgroundContainer.appendChild(line);
        }
        
        mainContainer.appendChild(backgroundContainer);
    }
    
    // Function to show confetti animation
    function showConfetti() {
        const confettiContainer = document.createElement('div');
        confettiContainer.classList.add('confetti-container');
        
        const colors = ['#00c853', '#3a7bd5', '#00d2ff', '#FFC107'];
        
        for (let i = 0; i < 80; i++) {
            const confetti = document.createElement('div');
            confetti.classList.add('confetti');
            confetti.style.left = `${Math.random() * 100}%`;
            confetti.style.animationDelay = `${Math.random() * 3}s`;
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confettiContainer.appendChild(confetti);
        }
        
        mainContainer.appendChild(confettiContainer);
        
        setTimeout(() => {
            confettiContainer.remove();
        }, 4000);
    }
    
    // Function to toggle between light and dark themes
    function toggleTheme() {
        isDarkTheme = !isDarkTheme;
        document.body.classList.toggle('light-theme');
        
        // Update theme icon
        if (isDarkTheme) {
            themeIcon.className = 'fas fa-moon';
        } else {
            themeIcon.className = 'fas fa-sun';
        }
        
        // Update chart if it exists
        if (priceChart && chartData.labels.length > 0) {
            createHistoricalChart(chartData);
        }
        
        // Save theme preference to localStorage
        localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
    }
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        toggleTheme();
    }

    // Function to display technical indicators
    function displayTechnicalIndicators(indicators) {
        // RSI
        const rsiValue = document.getElementById('rsi-value');
        const rsiSignal = document.getElementById('rsi-signal');
        rsiValue.textContent = indicators.rsi.value;
        rsiSignal.textContent = indicators.rsi.signal;
        rsiSignal.className = 'indicator-signal';
        rsiSignal.classList.add(indicators.rsi.trend);
        
        // MACD
        const macdValue = document.getElementById('macd-value');
        const macdSignal = document.getElementById('macd-signal');
        macdValue.textContent = indicators.macd.value;
        macdSignal.textContent = indicators.macd.signal;
        macdSignal.className = 'indicator-signal';
        macdSignal.classList.add(indicators.macd.trend);
        
        // Moving Average
        const maValue = document.getElementById('ma-value');
        const maSignal = document.getElementById('ma-signal');
        maValue.textContent = `${indicators.moving_average.ma50}/${indicators.moving_average.ma200}`;
        maSignal.textContent = indicators.moving_average.signal;
        maSignal.className = 'indicator-signal';
        maSignal.classList.add(indicators.moving_average.trend);
        
        // Volume
        const volumeValue = document.getElementById('volume-value');
        const volumeSignal = document.getElementById('volume-signal');
        volumeValue.textContent = formatVolume(indicators.volume.value);
        volumeSignal.textContent = indicators.volume.signal;
        volumeSignal.className = 'indicator-signal';
        volumeSignal.classList.add(indicators.volume.trend);
        
        // NEW: Bollinger Bands
        const bbValue = document.getElementById('bb-value');
        const bbSignal = document.getElementById('bb-signal');
        if (bbValue && bbSignal) {
            bbValue.textContent = `${indicators.bollinger_bands.lower}/${indicators.bollinger_bands.upper}`;
            bbSignal.textContent = indicators.bollinger_bands.signal;
            bbSignal.className = 'indicator-signal';
            bbSignal.classList.add(indicators.bollinger_bands.trend);
        }
        
        // NEW: Money Flow Index
        const mfiValue = document.getElementById('mfi-value');
        const mfiSignal = document.getElementById('mfi-signal');
        if (mfiValue && mfiSignal) {
            mfiValue.textContent = indicators.money_flow_index.value;
            mfiSignal.textContent = indicators.money_flow_index.signal;
            mfiSignal.className = 'indicator-signal';
            mfiSignal.classList.add(indicators.money_flow_index.trend);
        }
        
        // NEW: Risk Metrics
        const riskVolatility = document.getElementById('risk-volatility');
        const riskSharpe = document.getElementById('risk-sharpe');
        if (riskVolatility && riskSharpe) {
            riskVolatility.textContent = `${indicators.risk_metrics.volatility}% (${indicators.risk_metrics.volatility_level})`;
            riskSharpe.textContent = indicators.risk_metrics.sharpe_ratio;
            
            // Set volatility level class
            riskVolatility.className = 'risk-value';
            if (indicators.risk_metrics.volatility_level === 'High') {
                riskVolatility.classList.add('high-risk');
            } else if (indicators.risk_metrics.volatility_level === 'Medium') {
                riskVolatility.classList.add('medium-risk');
            } else {
                riskVolatility.classList.add('low-risk');
            }
            
            // Set Sharpe ratio class
            riskSharpe.className = 'risk-value';
            if (indicators.risk_metrics.sharpe_ratio > 1) {
                riskSharpe.classList.add('good-metric');
            } else if (indicators.risk_metrics.sharpe_ratio > 0) {
                riskSharpe.classList.add('neutral-metric');
            } else {
                riskSharpe.classList.add('bad-metric');
            }
        }
    }
    
    // NEW: Function to display news sentiment
    function displayNewsSentiment(sentiment) {
        // Set sentiment meter position (from -100% to 100%)
        const sentimentScore = sentiment.sentiment_score;
        const sentimentPercent = ((sentimentScore + 1) / 2) * 100; // Convert -1...1 to 0...100
        
        const sentimentIndicator = document.getElementById('sentiment-indicator');
        if (sentimentIndicator) {
            sentimentIndicator.style.left = `${sentimentPercent}%`;
        }
        
        // Update sentiment label
        const sentimentLabel = document.getElementById('sentiment-label');
        if (sentimentLabel) {
            sentimentLabel.textContent = sentiment.sentiment_label;
            sentimentLabel.className = 'sentiment-value';
            sentimentLabel.classList.add(sentiment.sentiment_label.toLowerCase());
        }
        
        // Display news items
        const newsContainer = document.getElementById('news-container');
        if (newsContainer) {
            // Clear previous news
            newsContainer.innerHTML = '';
            
            // Add news items
            sentiment.news_items.forEach(item => {
                const newsItem = document.createElement('div');
                newsItem.className = 'news-item';
                
                newsItem.innerHTML = `
                    <div class="news-date">${item.date}</div>
                    <div class="news-title">${item.headline}</div>
                    <span class="news-sentiment-tag ${item.sentiment}">${item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}</span>
                `;
                
                newsContainer.appendChild(newsItem);
            });
        }
    }

    // Helper function to format volume numbers
    function formatVolume(volume) {
        if (volume >= 1000000) {
            return (volume / 1000000).toFixed(1) + 'M';
        } else if (volume >= 1000) {
            return (volume / 1000).toFixed(1) + 'K';
        }
        return volume;
    }
}); 