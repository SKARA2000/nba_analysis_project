Predicting NBA Player Performance: From Moving Averages to Player Interactions

Machine Learning Term Project Outline  |  October 26, 2026
# **Purpose:** To develop and evaluate machine learning models that predict an NBA player's points scored in their next game using historical performance and information available before tipoff.
# Objectives:
- Establish baseline predictions using historical averages.
- Develop and compare regression and neural network models.
- Investigate how concept drift, including changes in player roles and team circumstances, affects prediction accuracy.
- Improve promising models through feature engineering and hyperparameter tuning.
- Evaluate whether increased model complexity provides meaningful improvements in predictive performance.
# **Evaluation Standards:** Compare models using consistent chronological training and testing procedures, with Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) as primary performance metrics.
# Problem and motivation
How well can we predict an NBA player’s next game using information available before tipoff? Player scoring is noisy, and a player’s role changes with teammates, injuries, opponents, and minutes played. We want to understand how much of that variation can be captured by simple historical estimates, when additional features help, and whether models that account for interactions between players are worth the extra complexity.

Our initial target is points scored in the next game. We will start with predictions of expected points, then investigate predictive distributions and the probability of exceeding different scoring thresholds. The project is organized as a sequence of questions, with each stage building on the previous one.
# Data and prediction setup
We plan to use historical player-game box scores from the NBA Game Logs dataset on Hugging Face and/or the NBA Database on Kaggle, spanning multiple seasons. These provide player outcomes, minutes, dates, teams, and opponents; additional game context can be constructed from earlier games. The unit of analysis for the initial models is a player-game observation. For the interaction model, we will organize observations by matchup.

Each prediction will use only information that would have been known before that game, some features may include: previous player and team statistics, home/away status, rest, and an expected rotation inferred from prior games. Injury-report history may be added if it can be retrieved consistently. We will evaluate later games using chronological splits, rather than randomizing observations across time.  everything will be tested through a walk through validation, and no individual  out come will be determined from data during or past that game.
# Questions and models, in increasing complexity
1. How far do simple historical estimates get us?

We will compare previous-season averages, season-to-date averages, and moving averages over the last 3, 5, and 10 games. We will also test exponentially weighted averages. This establishes whether recent form adds predictive value over a longer history and how quickly a useful estimate should adapt when a player’s role changes.

2. Can linear methods improve on those averages? Which features actually matter?

We will build linear and regularized regression models using recent performance, minutes, home/away status, rest, and opponent or team statistics. History-window length and weighting schemes can be selected as hyperparameters, or the model can learn coefficients over lagged statistics directly. We will test feature engineering, feature selection, regularization, and dimensionality reduction where appropriate. Beyond error rates, we want to examine which regressors are useful and whether their effects are stable across seasons and player roles.

3. Should we predict totals directly or normalize by minutes?

A player’s scoring rate and playing time reflect different sources of variation. We will compare direct points prediction against a factorized approach that predicts minutes and points per minute, then combines them. Because both terms are uncertain, we will evaluate not only point estimates but also how uncertainty propagates into the final points distribution. Actual minutes from the target game will never be used as a prediction input.

4. Can learned player representations improve prediction?

Instead of treating player identity as a fixed label or fitting a separate model for every player, we can learn a compact embedding for each player and combine it with time-varying features. We will compare an embedding-based neural model with the linear baselines and examine whether it generalizes when a player changes role or team.

5. Does modeling the other players in a matchup help?

A more ambitious model would represent each game using the expected rotation of both teams, estimated from recent appearances and minutes. An attention layer could update each player’s embedding using the identities and recent performance of teammates and opponents, before predicting points for each player. We will compare this against a simpler model that uses pooled roster features, to test whether attention itself contributes useful information.

6. What can we learn from the representations?

If the embedding model is successful, we would like to visualize players in a two-dimensional projection (for example, PCA or T-SNE), compare styles among well-known players, and investigate whether directions in the embedding space align with interpretable traits such as scoring, playmaking, or defensive statistics. These relationships would be tested against observed features rather than assumed from a visually appealing plot.
# Evaluation and division of work
All methods will be compared on the same held-out player-games using MAE and RMSE. For distributional predictions, we will examine likelihood or other proper scoring rules, prediction-interval coverage, and calibration at scoring thresholds. We will also break down performance by season stage, recent playing-time changes, and player history length to see where each approach succeeds or fails.

This progression gives team members several independent but connected directions: data construction and benchmark forecasting; linear modeling, feature analysis, and dimensionality reduction; uncertainty and minutes modeling; and representation learning and attention. The simpler methods form useful results on their own and directly apply course material, while the neural models leave room for members interested in a deeper extension. The final report will emphasize what each increase in complexity buys us, including cases where it buys nothing.
# Preliminary references and data
NBA Game Logs, Hugging Face: https://huggingface.co/datasets/cbratkovics/nba-game-logs

NBA Database, Kaggle: https://www.kaggle.com/datasets/wyattowalsh/basketball

Vaswani et al. (2017), “Attention Is All You Need”: https://arxiv.org/abs/1706.03762

Lee et al. (2019), “Set Transformer”: https://proceedings.mlr.press/v97/lee19d.html

Guan et al. (2023), “NBA2Vec”: https://arxiv.org/abs/2302.13386
