# A Hybrid Physics–Machine Learning Framework for Polymer Property Prediction

We placed 5th in the NeurIPS Open Polymer Prediction 2025 competition with an approach that mixed machine learning and physical insight. The task was to predict five polymer properties: glass transition temperature (Tg), fractional free volume (FFV), thermal conductivity (Tc), density, and radius of gyration (Rg). Instead of relying on one big model, we built things step by step so that some predictions could support the others.

We started with FFV and Tc. These properties had more data behind them and were strongly linked to the rest, so they felt like a natural starting point. By training ensembles of models, we got predictions stable enough to feed into the later stages. This way, we could take advantage of the fact that polymer properties are connected, instead of treating each one as if it lived in isolation.

Tg turned out to be the real challenge. The data for it were sparse, and plain ML models often struggled to generalize. Our breakthrough came when we blended those ML predictions with a physics-based model using its mathematical expressions. That combination gave Tg predictions that not only made physical sense but also ranked higher on the leaderboard. Without this step, we probably wouldn’t have been at the top 5.

We applied the same idea to density and Rg. Predictions from FFV and Tc were carried forward, and we adjusted outputs to handle distribution shifts and reduce bias. Although it did not contribute to the increase in results, it helped maintain stability.

Our feature extraction method was also helpful to get a great score. Alongside structural features capturing chemical patterns, we added molecular descriptors and indices related to polymer behavior. After some straightforward cleaning and scaling, these features gave the models a solid and balanced picture of the polymers.

Validation was another key factor. We used grouped cross-validation to avoid leakage and leaned on out-of-fold predictions to guide tuning. That setup gave us feedback we could trust and made sure improvements during training actually carried over to the final scores.

All in all, our success came from combining physics with ML, structuring the predictions hierarchically, and keeping validation tight. This mix proved strong enough to carry us into the top tier of the competition.