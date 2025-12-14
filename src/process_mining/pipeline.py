"""
Pipeline Process Mining
=======================

Pipeline complet pour l'analyse des traces d'urgences avec PM4Py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class ProcessMiningPipeline:
    """
    Pipeline complet de Process Mining avec PM4Py.
    
    Remplace Celonis Enterprise pour l'analyse des processus urgences.
    
    Fonctionnalités:
        - Import des event logs (CSV, XES)
        - Découverte de processus
        - Conformance checking
        - Analyse de performance
        - KPIs calibrés
    
    Example:
        >>> pipeline = ProcessMiningPipeline('traces_urgences.csv')
        >>> process_model = pipeline.discover_process()
        >>> kpis = pipeline.compute_kpis()
    """
    
    def __init__(
        self,
        event_log_path: str = None,
        case_id: str = 'case:concept:name',
        activity: str = 'concept:name',
        timestamp: str = 'time:timestamp',
        resource: str = 'org:resource'
    ):
        """
        Args:
            event_log_path: Chemin vers le fichier event log
            case_id: Nom de la colonne case ID
            activity: Nom de la colonne activité
            timestamp: Nom de la colonne timestamp
            resource: Nom de la colonne ressource
        """
        self.case_id = case_id
        self.activity = activity
        self.timestamp = timestamp
        self.resource = resource
        
        self.event_log = None
        self.df_log = None
        
        if event_log_path:
            self.load_event_log(event_log_path)
    
    def load_event_log(
        self,
        path: str,
        separator: str = ','
    ) -> 'ProcessMiningPipeline':
        """
        Charge un event log depuis un fichier.
        
        Supporte CSV et XES.
        
        Args:
            path: Chemin du fichier
            separator: Séparateur CSV
            
        Returns:
            self
        """
        import pm4py
        
        path = Path(path)
        
        if path.suffix.lower() == '.xes':
            self.event_log = pm4py.read_xes(str(path))
        elif path.suffix.lower() in ['.csv', '.tsv']:
            df = pd.read_csv(path, sep=separator)
            
            # Renommer les colonnes si nécessaire
            column_mapping = {}
            if self.case_id != 'case:concept:name' and self.case_id in df.columns:
                column_mapping[self.case_id] = 'case:concept:name'
            if self.activity != 'concept:name' and self.activity in df.columns:
                column_mapping[self.activity] = 'concept:name'
            if self.timestamp != 'time:timestamp' and self.timestamp in df.columns:
                column_mapping[self.timestamp] = 'time:timestamp'
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Convertir timestamp
            if 'time:timestamp' in df.columns:
                df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
            
            self.df_log = df
            self.event_log = pm4py.convert_to_event_log(df)
        else:
            raise ValueError(f"Format non supporté: {path.suffix}")
        
        print(f"  ✓ Event log chargé: {len(self.event_log)} traces")
        
        return self
    
    def get_dataframe(self) -> pd.DataFrame:
        """Retourne le log sous forme de DataFrame."""
        if self.df_log is not None:
            return self.df_log
        
        import pm4py
        return pm4py.convert_to_dataframe(self.event_log)
    
    def discover_process(
        self,
        algorithm: str = 'inductive'
    ) -> Any:
        """
        Découvre le modèle de processus.
        
        Args:
            algorithm: 'alpha', 'inductive', ou 'heuristic'
            
        Returns:
            Modèle de processus (Petri net ou Process tree)
        """
        import pm4py
        
        if self.event_log is None:
            raise ValueError("Event log non chargé")
        
        if algorithm == 'alpha':
            net, im, fm = pm4py.discover_petri_net_alpha(self.event_log)
            return {'net': net, 'initial_marking': im, 'final_marking': fm}
        
        elif algorithm == 'inductive':
            process_tree = pm4py.discover_process_tree_inductive(self.event_log)
            return process_tree
        
        elif algorithm == 'heuristic':
            net, im, fm = pm4py.discover_petri_net_heuristics(self.event_log)
            return {'net': net, 'initial_marking': im, 'final_marking': fm}
        
        else:
            raise ValueError(f"Algorithme inconnu: {algorithm}")
    
    def check_conformance(
        self,
        model: Any = None,
        algorithm: str = 'token_replay'
    ) -> Dict[str, float]:
        """
        Vérifie la conformance du log par rapport au modèle.
        
        Args:
            model: Modèle de processus (découvert si None)
            algorithm: 'token_replay' ou 'alignments'
            
        Returns:
            Dict avec fitness, precision, etc.
        """
        import pm4py
        
        if self.event_log is None:
            raise ValueError("Event log non chargé")
        
        if model is None:
            model = self.discover_process('inductive')
        
        # Convertir process tree en Petri net si nécessaire
        if hasattr(model, 'children'):  # Process tree
            net, im, fm = pm4py.convert_to_petri_net(model)
        else:
            net = model['net']
            im = model['initial_marking']
            fm = model['final_marking']
        
        if algorithm == 'token_replay':
            fitness = pm4py.fitness_token_based_replay(
                self.event_log, net, im, fm
            )
        else:
            fitness = pm4py.fitness_alignments(
                self.event_log, net, im, fm
            )
        
        # Precision
        precision = pm4py.precision_token_based_replay(
            self.event_log, net, im, fm
        )
        
        return {
            'fitness': fitness.get('average_trace_fitness', 0),
            'precision': precision,
            'f1': 2 * fitness.get('average_trace_fitness', 0) * precision / 
                  (fitness.get('average_trace_fitness', 0) + precision + 1e-6)
        }
    
    def compute_kpis(self) -> Dict[str, float]:
        """
        Calcule les KPIs du service d'urgences.
        
        KPIs:
            - dms_median: Durée médiane de séjour
            - dms_mean: Durée moyenne de séjour
            - dms_p95: 95e percentile DMS
            - attente_triage: Temps d'attente avant triage
            - attente_consultation: Temps d'attente avant consultation
            - n_traces: Nombre de parcours patients
            - n_activities: Nombre d'activités distinctes
            
        Returns:
            Dict des KPIs
        """
        df = self.get_dataframe()
        
        if df.empty:
            return {}
        
        kpis = {}
        
        # Nombre de traces
        case_col = 'case:concept:name'
        kpis['n_traces'] = df[case_col].nunique()
        
        # Activités distinctes
        activity_col = 'concept:name'
        kpis['n_activities'] = df[activity_col].nunique()
        
        # Calcul DMS par case
        timestamp_col = 'time:timestamp'
        if timestamp_col in df.columns:
            case_times = df.groupby(case_col)[timestamp_col].agg(['min', 'max'])
            dms = (case_times['max'] - case_times['min']).dt.total_seconds() / 60
            
            kpis['dms_mean'] = float(dms.mean())
            kpis['dms_median'] = float(dms.median())
            kpis['dms_p95'] = float(dms.quantile(0.95))
            kpis['dms_std'] = float(dms.std())
        
        # Temps d'attente par activité
        activities_attente = ['Triage', 'Consultation', 'Scanner', 'Radio']
        
        for activity in activities_attente:
            activity_df = df[df[activity_col].str.contains(activity, case=False, na=False)]
            if not activity_df.empty and timestamp_col in df.columns:
                # Calculer temps depuis arrivée
                first_times = df.groupby(case_col)[timestamp_col].min()
                activity_times = activity_df.groupby(case_col)[timestamp_col].min()
                
                common_cases = first_times.index.intersection(activity_times.index)
                if len(common_cases) > 0:
                    wait_times = (activity_times[common_cases] - first_times[common_cases]).dt.total_seconds() / 60
                    kpis[f'attente_{activity.lower()}_median'] = float(wait_times.median())
        
        # Distribution des parcours (variantes)
        traces = df.groupby(case_col)[activity_col].apply(lambda x: '→'.join(x))
        kpis['n_variants'] = traces.nunique()
        kpis['top_variant_pct'] = float(traces.value_counts().iloc[0] / len(traces) * 100)
        
        return kpis
    
    def get_case_duration_distribution(self) -> pd.Series:
        """Retourne la distribution des durées de cas."""
        df = self.get_dataframe()
        
        case_col = 'case:concept:name'
        timestamp_col = 'time:timestamp'
        
        case_times = df.groupby(case_col)[timestamp_col].agg(['min', 'max'])
        return (case_times['max'] - case_times['min']).dt.total_seconds() / 60
    
    def get_activity_statistics(self) -> pd.DataFrame:
        """Retourne les statistiques par activité."""
        df = self.get_dataframe()
        
        activity_col = 'concept:name'
        
        stats = df.groupby(activity_col).agg({
            'case:concept:name': 'count'
        }).rename(columns={'case:concept:name': 'count'})
        
        stats['percentage'] = stats['count'] / stats['count'].sum() * 100
        
        return stats.sort_values('count', ascending=False)
    
    def export_distributions(
        self,
        output_path: str = 'distributions.json'
    ) -> Dict[str, Any]:
        """
        Exporte les distributions calibrées pour la simulation.
        
        Args:
            output_path: Fichier de sortie
            
        Returns:
            Dict des distributions
        """
        import json
        from scipy import stats
        
        df = self.get_dataframe()
        distributions = {}
        
        # Durées par activité
        activity_col = 'concept:name'
        timestamp_col = 'time:timestamp'
        
        # Grouper par case et calculer durées entre activités
        df_sorted = df.sort_values([self.case_id, timestamp_col])
        df_sorted['duration'] = df_sorted.groupby(self.case_id)[timestamp_col].diff().dt.total_seconds() / 60
        
        for activity in df[activity_col].unique():
            activity_durations = df_sorted[df_sorted[activity_col] == activity]['duration'].dropna()
            
            if len(activity_durations) > 10:
                # Fit lognormal
                shape, loc, scale = stats.lognorm.fit(activity_durations.clip(lower=0.1))
                
                distributions[activity.lower().replace(' ', '_')] = {
                    'type': 'lognormal',
                    'params': {
                        'mean': float(np.log(scale)),
                        'sigma': float(shape)
                    },
                    'stats': {
                        'median': float(activity_durations.median()),
                        'mean': float(activity_durations.mean()),
                        'std': float(activity_durations.std()),
                        'n': len(activity_durations)
                    }
                }
        
        # Taux d'arrivée par heure
        if timestamp_col in df.columns:
            df['hour'] = df[timestamp_col].dt.hour
            arrivals = df.groupby([self.case_id, 'hour']).first().reset_index()
            hourly_counts = arrivals.groupby('hour').size()
            
            n_days = (df[timestamp_col].max() - df[timestamp_col].min()).days or 1
            
            distributions['lambda_par_heure'] = {
                heure: float(count / n_days)
                for heure, count in hourly_counts.items()
            }
        
        # Sauvegarder
        with open(output_path, 'w') as f:
            json.dump(distributions, f, indent=2)
        
        print(f"  ✓ Distributions exportées: {output_path}")
        
        return distributions
    
    def visualize_process(
        self,
        model: Any = None,
        output_path: str = None
    ) -> None:
        """
        Visualise le modèle de processus.
        
        Args:
            model: Modèle à visualiser
            output_path: Chemin de sortie (affichage si None)
        """
        import pm4py
        
        if model is None:
            model = self.discover_process('inductive')
        
        if hasattr(model, 'children'):  # Process tree
            pm4py.view_process_tree(model)
            if output_path:
                pm4py.save_vis_process_tree(model, output_path)
        else:
            net = model['net']
            im = model['initial_marking']
            fm = model['final_marking']
            pm4py.view_petri_net(net, im, fm)
            if output_path:
                pm4py.save_vis_petri_net(net, im, fm, output_path)
