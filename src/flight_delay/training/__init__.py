"""Trening modelu: split czasowy, baseline, ewaluacja, XGBoost.

Faza A: sklearn/XGBoost wywoływane wprost. Faza B: te same funkcje owijane
w taski Airflow i logowane do MLflow - logika zostaje tutaj, orkiestracja na zewnątrz.
"""
