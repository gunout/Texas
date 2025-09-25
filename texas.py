# texas_real_estate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TexasRealEstateAnalyzer:
    def __init__(self, region_name):
        self.region = region_name
        self.colors = ['#BF0A30', '#002868', '#666666', '#008751', '#FFA300', 
                      '#8B4513', '#228B22', '#FFD700', '#8A2BE2', '#DC143C']
        
        self.start_year = 2002
        self.end_year = 2025
        
        # Configuration spécifique à chaque région du Texas
        self.config = self._get_region_config()
        
    def _get_region_config(self):
        """Retourne la configuration spécifique pour chaque région du Texas"""
        configs = {
            "Dallas-Fort Worth": {
                "population_base": 7600000,
                "budget_base": 9800,
                "type": "corporate_tech",
                "specialites": ["technologie", "finance", "corporate", "logistique", "defense"],
                "prix_m2_base": 3200,
                "segment_immobilier": "corporate_affordable",
                "currency": "USD",
                "major_cities": ["Dallas", "Fort Worth", "Arlington", "Plano"]
            },
            "Houston Metro": {
                "population_base": 7300000,
                "budget_base": 9200,
                "type": "energy_medical",
                "specialites": ["énergie", "pétrole", "médecine", "port", "aérospatial"],
                "prix_m2_base": 2800,
                "segment_immobilier": "energy_driven",
                "currency": "USD",
                "major_cities": ["Houston", "The Woodlands", "Sugar Land", "Pearland"]
            },
            "Austin Area": {
                "population_base": 2300000,
                "budget_base": 4800,
                "type": "tech_innovation",
                "specialites": ["technologie", "innovation", "musique", "éducation", "startups"],
                "prix_m2_base": 4500,
                "segment_immobilier": "tech_boom",
                "currency": "USD",
                "major_cities": ["Austin", "Round Rock", "Cedar Park", "San Marcos"]
            },
            "San Antonio": {
                "population_base": 2600000,
                "budget_base": 3800,
                "type": "military_tourism",
                "specialites": ["militaire", "tourisme", "santé", "éducation", "culture"],
                "prix_m2_base": 2200,
                "segment_immobilier": "affordable_growth",
                "currency": "USD",
                "major_cities": ["San Antonio", "New Braunfels", "Schertz", "Converse"]
            },
            "El Paso Area": {
                "population_base": 850000,
                "budget_base": 1800,
                "type": "border_manufacturing",
                "specialites": ["manufacturing", "commerce_frontalier", "defense", "logistique", "services"],
                "prix_m2_base": 1500,
                "segment_immobilier": "border_affordable",
                "currency": "USD",
                "major_cities": ["El Paso", "Socorro", "Horizon City"]
            },
            "Rio Grande Valley": {
                "population_base": 1400000,
                "budget_base": 2200,
                "type": "agricultural_border",
                "specialites": ["agriculture", "commerce_frontalier", "tourisme", "santé", "éducation"],
                "prix_m2_base": 1200,
                "segment_immobilier": "rural_affordable",
                "currency": "USD",
                "major_cities": ["McAllen", "Brownsville", "Edinburg", "Harlingen"]
            },
            "West Texas": {
                "population_base": 600000,
                "budget_base": 1500,
                "type": "energy_agricultural",
                "specialites": ["énergie", "pétrole", "agriculture", "élevage", "éolien"],
                "prix_m2_base": 1800,
                "segment_immobilier": "rural_energy",
                "currency": "USD",
                "major_cities": ["Midland", "Odessa", "Lubbock", "Amarillo"]
            },
            "Central Texas": {
                "population_base": 1200000,
                "budget_base": 2500,
                "type": "mixed_agricultural",
                "specialites": ["agriculture", "manufacturing", "éducation", "services", "tourisme_rural"],
                "prix_m2_base": 2000,
                "segment_immobilier": "rural_mixed",
                "currency": "USD",
                "major_cities": ["Waco", "Temple", "Killeen", "College Station"]
            },
            # Configuration par défaut
            "default": {
                "population_base": 1000000,
                "budget_base": 2000,
                "type": "mixed_development",
                "specialites": ["residentiel", "commerce_local", "services"],
                "prix_m2_base": 2500,
                "segment_immobilier": "mixed",
                "currency": "USD",
                "major_cities": ["Multiple cities"]
            }
        }
        
        return configs.get(self.region, configs["default"])
    
    def generate_financial_data(self):
        """Génère des données financières et immobilières pour la région du Texas"""
        print(f"🤠 Génération des données financières et immobilières pour {self.region}, Texas...")
        
        # Créer une base de données annuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Year': [date.year for date in dates]}
        
        # Données démographiques
        data['Population'] = self._simulate_population(dates)
        data['Households'] = self._simulate_households(dates)
        data['Median_Income'] = self._simulate_median_income(dates)
        
        # Recettes régionales (en millions de dollars)
        data['Total_Revenue'] = self._simulate_total_revenue(dates)
        data['Property_Tax_Revenue'] = self._simulate_property_tax_revenue(dates)
        data['State_Federal_Funding'] = self._simulate_government_funding(dates)
        data['Business_Tax_Revenue'] = self._simulate_business_tax_revenue(dates)
        data['Energy_Revenue'] = self._simulate_energy_revenue(dates)
        data['Other_Revenue'] = self._simulate_other_revenue(dates)
        
        # Dépenses régionales
        data['Total_Expenses'] = self._simulate_total_expenses(dates)
        data['Infrastructure_Expenses'] = self._simulate_infrastructure_expenses(dates)
        data['Public_Services_Expenses'] = self._simulate_public_services_expenses(dates)
        data['Education_Expenses'] = self._simulate_education_expenses(dates)
        data['Healthcare_Expenses'] = self._simulate_healthcare_expenses(dates)
        
        # Indicateurs financiers
        data['Budget_Surplus_Deficit'] = self._simulate_budget_balance(dates)
        data['Regional_Debt'] = self._simulate_regional_debt(dates)
        data['Debt_to_Revenue_Ratio'] = self._simulate_debt_ratio(dates)
        
        # Données immobilières (spécifiques au Texas)
        data['Median_Home_Price'] = self._simulate_median_home_price(dates)
        data['Price_per_Sqft'] = self._simulate_price_per_sqft(dates)
        data['Home_Sales_Volume'] = self._simulate_home_sales(dates)
        data['New_Construction_Permits'] = self._simulate_construction_permits(dates)
        data['Rental_Vacancy_Rate'] = self._simulate_vacancy_rate(dates)
        data['Average_Rent'] = self._simulate_average_rent(dates)
        
        # Investissements spécifiques adaptés au Texas
        data['Energy_Investment'] = self._simulate_energy_investment(dates)
        data['Tech_Investment'] = self._simulate_tech_investment(dates)
        data['Infrastructure_Investment'] = self._simulate_infrastructure_investment(dates)
        data['Housing_Development_Investment'] = self._simulate_housing_investment(dates)
        data['Manufacturing_Investment'] = self._simulate_manufacturing_investment(dates)
        data['Agricultural_Investment'] = self._simulate_agricultural_investment(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances spécifiques au marché texan
        self._add_texas_trends(df)
        
        return df
    
    def _simulate_population(self, dates):
        """Simule la population de la région"""
        base_population = self.config["population_base"]
        
        population = []
        for i, date in enumerate(dates):
            # Croissance démographique texane (très forte croissance)
            if self.config["type"] == "tech_innovation":
                growth_rate = 0.028  # Très forte croissance à Austin
            elif self.config["type"] == "corporate_tech":
                growth_rate = 0.022  # Forte croissance à DFW
            elif self.config["type"] == "energy_medical":
                growth_rate = 0.020  # Croissance forte à Houston
            else:
                growth_rate = 0.015  # Croissance modérée ailleurs
                
            growth = 1 + growth_rate * i
            population.append(base_population * growth)
        
        return population
    
    def _simulate_households(self, dates):
        """Simule le nombre de ménages"""
        base_households = self.config["population_base"] / 2.7  # Taille moyenne des ménages au Texas
        
        households = []
        for i, date in enumerate(dates):
            growth = 1 + 0.016 * i
            households.append(base_households * growth)
        
        return households
    
    def _simulate_median_income(self, dates):
        """Simule le revenu médian"""
        # Revenu médian de base selon la région
        if self.config["type"] == "tech_innovation":
            base_income = 85000
        elif self.config["type"] == "corporate_tech":
            base_income = 75000
        elif self.config["type"] == "energy_medical":
            base_income = 70000
        else:
            base_income = 55000
        
        incomes = []
        for i, date in enumerate(dates):
            year = date.year
            # Croissance du revenu avec des variations
            if 2002 <= year <= 2008:
                growth = 1 + 0.040 * (year - 2002)  # Forte croissance énergie
            elif 2009 <= year <= 2010:
                growth = 1 - 0.020 * (year - 2009)  # Légère baisse crise
            elif 2011 <= year <= 2014:
                growth = 1 + 0.045 * (year - 2011)  # Boom énergie
            elif 2015 <= year <= 2016:
                growth = 1 - 0.015  # Baisse prix pétrole
            elif 2017 <= year <= 2019:
                growth = 1 + 0.038 * (year - 2017)
            elif 2020 <= year <= 2021:
                growth = 1 - 0.010  # Impact COVID modéré
            else:
                growth = 1 + 0.042 * (year - 2022)
            
            noise = np.random.normal(1, 0.05)
            incomes.append(base_income * growth * noise)
        
        return incomes
    
    def _simulate_total_revenue(self, dates):
        """Simule les recettes totales de la région"""
        base_revenue = self.config["budget_base"]
        
        revenue = []
        for i, date in enumerate(dates):
            # Croissance économique texane
            if self.config["type"] == "tech_innovation":
                growth_rate = 0.065  # Croissance explosive à Austin
            elif self.config["type"] == "corporate_tech":
                growth_rate = 0.055  # Croissance forte à DFW
            elif self.config["type"] == "energy_medical":
                growth_rate = 0.050  # Croissance variable à Houston
            else:
                growth_rate = 0.042  # Croissance moyenne
                
            growth = 1 + growth_rate * i
            noise = np.random.normal(1, 0.12)  # Plus volatile dû à l'énergie
            revenue.append(base_revenue * growth * noise)
        
        return revenue
    
    def _simulate_property_tax_revenue(self, dates):
        """Simule les recettes de taxe foncière (pas de taxe sur le revenu au Texas)"""
        base_tax = self.config["budget_base"] * 0.45  # Très important au Texas
        
        tax_revenue = []
        for i, date in enumerate(dates):
            growth = 1 + 0.035 * i
            noise = np.random.normal(1, 0.08)
            tax_revenue.append(base_tax * growth * noise)
        
        return tax_revenue
    
    def _simulate_government_funding(self, dates):
        """Simule le financement étatique et fédéral"""
        base_funding = self.config["budget_base"] * 0.20
        
        funding = []
        for i, date in enumerate(dates):
            year = date.year
            if year >= 2010:
                increase = 1 + 0.012 * (year - 2010)
            else:
                increase = 1
            
            noise = np.random.normal(1, 0.08)
            funding.append(base_funding * increase * noise)
        
        return funding
    
    def _simulate_business_tax_revenue(self, dates):
        """Simule les recettes fiscales des entreprises"""
        base_business_tax = self.config["budget_base"] * 0.18
        
        business_tax = []
        for i, date in enumerate(dates):
            multiplier = 1.4 if "technologie" in self.config["specialites"] else 1.2 if "énergie" in self.config["specialites"] else 1.0
            
            growth = 1 + 0.048 * i
            noise = np.random.normal(1, 0.15)
            business_tax.append(base_business_tax * growth * multiplier * noise)
        
        return business_tax
    
    def _simulate_energy_revenue(self, dates):
        """Simule les recettes énergétiques (spécifique au Texas)"""
        base_energy = self.config["budget_base"] * 0.12
        
        multiplier = 2.5 if "énergie" in self.config["specialites"] else 0.3
        
        energy_revenue = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Volatilité selon les prix de l'énergie
            if 2002 <= year <= 2008:
                energy_multiplier = 1 + 0.15 * (year - 2002)  # Boom pétrole
            elif 2009 <= year <= 2010:
                energy_multiplier = 0.70  # Crise financière
            elif 2011 <= year <= 2014:
                energy_multiplier = 1 + 0.12 * (year - 2011)  # Reprise
            elif 2015 <= year <= 2016:
                energy_multiplier = 0.60  # Effondrement prix pétrole
            elif 2017 <= year <= 2019:
                energy_multiplier = 1 + 0.08 * (year - 2017)  # Reprise modérée
            elif 2020 <= year <= 2021:
                energy_multiplier = 0.75  # COVID + crise pétrole
            else:
                energy_multiplier = 1 + 0.10 * (year - 2022)  # Reprise forte
            
            growth = 1 + 0.030 * i
            noise = np.random.normal(1, 0.25)  # Très volatile
            energy_revenue.append(base_energy * growth * energy_multiplier * multiplier * noise)
        
        return energy_revenue
    
    def _simulate_other_revenue(self, dates):
        """Simule les autres recettes"""
        base_other = self.config["budget_base"] * 0.05
        
        other_revenue = []
        for i, date in enumerate(dates):
            growth = 1 + 0.028 * i
            noise = np.random.normal(1, 0.10)
            other_revenue.append(base_other * growth * noise)
        
        return other_revenue
    
    def _simulate_total_expenses(self, dates):
        """Simule les dépenses totales"""
        base_expenses = self.config["budget_base"] * 0.88  # Texas a des dépenses plus faibles
        
        expenses = []
        for i, date in enumerate(dates):
            growth = 1 + 0.038 * i
            noise = np.random.normal(1, 0.06)
            expenses.append(base_expenses * growth * noise)
        
        return expenses
    
    def _simulate_infrastructure_expenses(self, dates):
        """Simule les dépenses d'infrastructure"""
        base_infra = self.config["budget_base"] * 0.20
        
        infra_expenses = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2005, 2013, 2018, 2023]:
                multiplier = 1.8  # Années de grands projets
            else:
                multiplier = 1.0
            
            growth = 1 + 0.040 * i
            noise = np.random.normal(1, 0.18)
            infra_expenses.append(base_infra * growth * multiplier * noise)
        
        return infra_expenses
    
    def _simulate_public_services_expenses(self, dates):
        """Simule les dépenses de services publics"""
        base_services = self.config["budget_base"] * 0.25
        
        services = []
        for i, date in enumerate(dates):
            growth = 1 + 0.032 * i
            noise = np.random.normal(1, 0.04)
            services.append(base_services * growth * noise)
        
        return services
    
    def _simulate_education_expenses(self, dates):
        """Simule les dépenses éducatives"""
        base_education = self.config["budget_base"] * 0.28  # Important au Texas
        
        education = []
        for i, date in enumerate(dates):
            growth = 1 + 0.036 * i
            noise = np.random.normal(1, 0.05)
            education.append(base_education * growth * noise)
        
        return education
    
    def _simulate_healthcare_expenses(self, dates):
        """Simule les dépenses de santé"""
        base_healthcare = self.config["budget_base"] * 0.15
        
        healthcare = []
        for i, date in enumerate(dates):
            growth = 1 + 0.040 * i
            noise = np.random.normal(1, 0.06)
            healthcare.append(base_healthcare * growth * noise)
        
        return healthcare
    
    def _simulate_budget_balance(self, dates):
        """Simule le surplus/déficit budgétaire"""
        balance = []
        for i, date in enumerate(dates):
            base_balance = self.config["budget_base"] * 0.12  # Texas a généralement des surplus
            
            year = date.year
            if year >= 2010:
                improvement = 1 + 0.015 * (year - 2010)
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.15)
            balance.append(base_balance * improvement * noise)
        
        return balance
    
    def _simulate_regional_debt(self, dates):
        """Simule la dette régionale"""
        base_debt = self.config["budget_base"] * 0.45  # Dette faible au Texas
        
        debt = []
        for i, date in enumerate(dates):
            year = date.year
            if year >= 2012:
                reduction = 1 - 0.020 * (year - 2012)
            else:
                reduction = 1.0
            
            noise = np.random.normal(1, 0.07)
            debt.append(base_debt * reduction * noise)
        
        return debt
    
    def _simulate_debt_ratio(self, dates):
        """Simule le ratio d'endettement"""
        ratios = []
        for i, date in enumerate(dates):
            base_ratio = 0.40  # Ratio faible
            
            year = date.year
            if year >= 2012:
                improvement = 1 - 0.022 * (year - 2012)
            else:
                improvement = 1
            
            noise = np.random.normal(1, 0.06)
            ratios.append(base_ratio * improvement * noise)
        
        return ratios
    
    def _simulate_median_home_price(self, dates):
        """Simule le prix médian des maisons (spécifique au Texas)"""
        base_price = self.config["prix_m2_base"] * 200  # Maisons plus grandes au Texas
        
        prices = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance du marché immobilier texan
            if self.config["segment_immobilier"] == "tech_boom":
                growth_rate = 0.085  # Croissance explosive à Austin
            elif self.config["segment_immobilier"] == "corporate_affordable":
                growth_rate = 0.055  # Croissance forte à DFW
            elif self.config["segment_immobilier"] == "energy_driven":
                growth_rate = 0.048  # Croissance volatile à Houston
            else:
                growth_rate = 0.040  # Croissance modérée
            
            # Ajustements annuels basés sur des événements réels
            if 2002 <= year <= 2007:
                multiplier = 1 + 0.08 * (year - 2002)  # Boom pré-crise
            elif 2008 <= year <= 2009:
                multiplier = 0.90  # Légère baisse (Texas résilient)
            elif 2010 <= year <= 2014:
                multiplier = 1 + 0.10 * (year - 2010)  # Forte reprise
            elif 2015 <= year <= 2016:
                multiplier = 0.95  # Légère baisse énergie
            elif 2017 <= year <= 2019:
                multiplier = 1 + 0.07 * (year - 2017)  # Croissance forte
            elif 2020 <= year <= 2021:
                multiplier = 1.02  # Texas résilient pendant COVID
            else:
                multiplier = 1 + 0.09 * (year - 2022)  # Boom post-COVID
            
            growth = 1 + growth_rate * i
            noise = np.random.normal(1, 0.10)
            prices.append(base_price * growth * multiplier * noise)
        
        return prices
    
    def _simulate_price_per_sqft(self, dates):
        """Simule le prix au pied carré"""
        base_price_sqft = self.config["prix_m2_base"] / 10.764
        
        prices = []
        for i, date in enumerate(dates):
            median_price = self._simulate_median_home_price([date])[0]
            avg_home_size = 200 * 10.764  # 200m² en pieds carrés
            prices.append(median_price / avg_home_size)
        
        return prices
    
    def _simulate_home_sales(self, dates):
        """Simule le volume des ventes immobilières"""
        base_sales = self.config["population_base"] / 100  # Marché actif au Texas
        
        sales = []
        for i, date in enumerate(dates):
            year = date.year
            
            if 2002 <= year <= 2006:
                multiplier = 1 + 0.12 * (year - 2002)
            elif 2007 <= year <= 2009:
                multiplier = 0.80  # Baisse modérée
            elif 2010 <= year <= 2019:
                multiplier = 1 + 0.10 * (year - 2010)
            elif 2020 <= year <= 2021:
                multiplier = 0.90  # Légère baisse COVID
            else:
                multiplier = 1 + 0.11 * (year - 2022)
            
            growth = 1 + 0.018 * i
            noise = np.random.normal(1, 0.14)
            sales.append(base_sales * growth * multiplier * noise)
        
        return sales
    
    def _simulate_construction_permits(self, dates):
        """Simule les permis de construction"""
        base_permits = self.config["population_base"] / 400  # Forte construction au Texas
        
        permits = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year in [2005, 2013, 2018, 2022, 2024]:
                multiplier = 2.0  # Années de forte construction
            elif year in [2008, 2015, 2020]:
                multiplier = 0.7  # Années de ralentissement
            else:
                multiplier = 1.0
            
            growth = 1 + 0.025 * i
            noise = np.random.normal(1, 0.20)
            permits.append(base_permits * growth * multiplier * noise)
        
        return permits
    
    def _simulate_vacancy_rate(self, dates):
        """Simule le taux d'inoccupation locative"""
        base_vacancy = 6.0  # Plus élevé au Texas dû à plus de construction
        
        vacancies = []
        for i, date in enumerate(dates):
            year = date.year
            
            if 2002 <= year <= 2006:
                rate = base_vacancy - 0.8 * (year - 2002)
            elif 2007 <= year <= 2010:
                rate = base_vacancy + 1.5
            elif 2011 <= year <= 2019:
                rate = base_vacancy - 0.4 * (year - 2011)
            elif 2020 <= year <= 2021:
                rate = base_vacancy + 1.0
            else:
                rate = base_vacancy - 0.3 * (year - 2022)
            
            noise = np.random.normal(0, 0.4)
            vacancies.append(max(2.0, rate + noise))  # Minimum 2%
        
        return vacancies
    
    def _simulate_average_rent(self, dates):
        """Simule le loyer moyen"""
        base_rent = self.config["prix_m2_base"] / 40  # Loyer plus abordable au Texas
        
        rents = []
        for i, date in enumerate(dates):
            year = date.year
            
            if 2002 <= year <= 2007:
                growth = 1 + 0.035 * (year - 2002)
            elif 2008 <= year <= 2010:
                growth = 1 - 0.010 * (year - 2008)
            elif 2011 <= year <= 2019:
                growth = 1 + 0.040 * (year - 2011)
            elif 2020 <= year <= 2021:
                growth = 1 + 0.005  # Légère hausse
            else:
                growth = 1 + 0.045 * (year - 2022)
            
            noise = np.random.normal(1, 0.06)
            rents.append(base_rent * growth * noise)
        
        return rents
    
    def _simulate_energy_investment(self, dates):
        """Simule l'investissement énergétique (spécifique au Texas)"""
        base_investment = self.config["budget_base"] * 0.15
        
        multiplier = 3.0 if "énergie" in self.config["specialites"] else 0.4
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2003, 2008, 2012, 2017, 2021]:
                year_multiplier = 2.2
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.055 * i
            noise = np.random.normal(1, 0.30)  # Très volatile
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _simulate_tech_investment(self, dates):
        """Simule l'investissement technologique"""
        base_investment = self.config["budget_base"] * 0.12
        
        multiplier = 2.5 if "technologie" in self.config["specialites"] else 0.8
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2005, 2010, 2015, 2020, 2023]:
                year_multiplier = 1.9
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.070 * i
            noise = np.random.normal(1, 0.18)
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _simulate_infrastructure_investment(self, dates):
        """Simule l'investissement en infrastructure"""
        base_investment = self.config["budget_base"] * 0.20
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2004, 2011, 2016, 2022]:
                year_multiplier = 1.8
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.045 * i
            noise = np.random.normal(1, 0.15)
            investment.append(base_investment * growth * year_multiplier * noise)
        
        return investment
    
    def _simulate_housing_investment(self, dates):
        """Simule l'investissement dans le logement"""
        base_investment = self.config["budget_base"] * 0.18  # Important au Texas
        
        multiplier = 1.5  # Texas a une forte construction
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2006, 2013, 2019, 2024]:
                year_multiplier = 1.8
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.050 * i
            noise = np.random.normal(1, 0.20)
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _simulate_manufacturing_investment(self, dates):
        """Simule l'investissement manufacturier"""
        base_investment = self.config["budget_base"] * 0.10
        
        multiplier = 1.8 if "manufacturing" in self.config["specialites"] else 0.7
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2007, 2014, 2018, 2023]:
                year_multiplier = 1.7
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.038 * i
            noise = np.random.normal(1, 0.16)
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _simulate_agricultural_investment(self, dates):
        """Simule l'investissement agricole"""
        base_investment = self.config["budget_base"] * 0.08
        
        multiplier = 2.2 if "agriculture" in self.config["specialites"] else 0.9
        
        investment = []
        for i, date in enumerate(dates):
            year = date.year
            if year in [2009, 2012, 2017, 2021]:
                year_multiplier = 1.6
            else:
                year_multiplier = 1.0
            
            growth = 1 + 0.032 * i
            noise = np.random.normal(1, 0.19)
            investment.append(base_investment * growth * year_multiplier * multiplier * noise)
        
        return investment
    
    def _add_texas_trends(self, df):
        """Ajoute des tendances réalistes adaptées au marché texan"""
        for i, row in df.iterrows():
            year = row['Year']
            
            # Boom pétrolier (2003-2008, 2011-2014)
            if (2003 <= year <= 2008) or (2011 <= year <= 2014):
                if "énergie" in self.config["specialites"]:
                    df.loc[i, 'Energy_Revenue'] *= 1.8
                    df.loc[i, 'Median_Income'] *= 1.10
                    df.loc[i, 'Population'] *= 1.03
            
            # Crise financière (2008-2009) - impact modéré au Texas
            if 2008 <= year <= 2009:
                df.loc[i, 'Median_Home_Price'] *= 0.90
                df.loc[i, 'Home_Sales_Volume'] *= 0.80
                df.loc[i, 'Energy_Investment'] *= 0.70
            
            # Effondrement prix pétrole (2015-2016)
            if 2015 <= year <= 2016:
                if "énergie" in self.config["specialites"]:
                    df.loc[i, 'Energy_Revenue'] *= 0.40
                    df.loc[i, 'Median_Home_Price'] *= 0.92
                    df.loc[i, 'Population'] *= 0.99
            
            # Boom tech à Austin (2015-présent)
            if year >= 2015 and "technologie" in self.config["specialites"]:
                df.loc[i, 'Tech_Investment'] *= 2.2
                df.loc[i, 'Median_Home_Price'] *= 1.15
                df.loc[i, 'Population'] *= 1.04
            
            # Croissance démographique forte (constant)
            if year >= 2010:
                df.loc[i, 'Housing_Development_Investment'] *= 1.4
                df.loc[i, 'New_Construction_Permits'] *= 1.3
            
            # Impact COVID-19 (2020-2021)
            if 2020 <= year <= 2021:
                if year == 2020:
                    df.loc[i, 'Energy_Revenue'] *= 0.60
                    df.loc[i, 'Home_Sales_Volume'] *= 0.85
                else:
                    # Reprise forte au Texas
                    df.loc[i, 'Median_Home_Price'] *= 1.08
                    df.loc[i, 'Population'] *= 1.02
            
            # Boom post-COVID (2022-présent)
            if year >= 2022:
                df.loc[i, 'Tech_Investment'] *= 1.3
                df.loc[i, 'Manufacturing_Investment'] *= 1.4
                df.loc[i, 'Infrastructure_Investment'] *= 1.5
    
    def create_financial_analysis(self, df):
        """Crée une analyse complète des finances et de l'immobilier texan"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 28))
        
        # 1. Évolution des prix immobiliers
        ax1 = plt.subplot(5, 2, 1)
        self._plot_real_estate_prices(df, ax1)
        
        # 2. Activité immobilière
        ax2 = plt.subplot(5, 2, 2)
        self._plot_real_estate_activity(df, ax2)
        
        # 3. Évolution des recettes et dépenses
        ax3 = plt.subplot(5, 2, 3)
        self._plot_revenue_expenses(df, ax3)
        
        # 4. Structure des recettes (avec énergie)
        ax4 = plt.subplot(5, 2, 4)
        self._plot_revenue_structure(df, ax4)
        
        # 5. Marché locatif
        ax5 = plt.subplot(5, 2, 5)
        self._plot_rental_market(df, ax5)
        
        # 6. Investissements régionaux
        ax6 = plt.subplot(5, 2, 6)
        self._plot_regional_investments(df, ax6)
        
        # 7. Démographie et revenus
        ax7 = plt.subplot(5, 2, 7)
        self._plot_demography_income(df, ax7)
        
        # 8. Dette et équilibre budgétaire
        ax8 = plt.subplot(5, 2, 8)
        self._plot_debt_budget(df, ax8)
        
        # 9. Construction et développement
        ax9 = plt.subplot(5, 2, 9)
        self._plot_construction_development(df, ax9)
        
        # 10. Investissements sectoriels
        ax10 = plt.subplot(5, 2, 10)
        self._plot_sectorial_investments(df, ax10)
        
        plt.suptitle(f'Financial and Real Estate Analysis of {self.region}, Texas ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.region.replace(" ", "_").lower()}_texas_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_texas_insights(df)
    
    def _plot_real_estate_prices(self, df, ax):
        """Plot de l'évolution des prix immobiliers"""
        ax.plot(df['Year'], df['Median_Home_Price']/1000, label='Median Home Price', 
               linewidth=3, color='#BF0A30', alpha=0.8)
        
        ax.set_title('Median Home Price Evolution (Thousand $)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (Thousand $)')
        ax.grid(True, alpha=0.3)
        
        # Ajouter des annotations pour les événements marquants
        ax.annotate('Oil Boom', xy=(2006, df.loc[df['Year'] == 2006, 'Median_Home_Price'].values[0]/1000), 
                   xytext=(2006, df.loc[df['Year'] == 2006, 'Median_Home_Price'].values[0]/1000 * 0.8),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.annotate('Tech Boom', xy=(2018, df.loc[df['Year'] == 2018, 'Median_Home_Price'].values[0]/1000), 
                   xytext=(2018, df.loc[df['Year'] == 2018, 'Median_Home_Price'].values[0]/1000 * 1.4),
                   arrowprops=dict(arrowstyle='->', color='green'))
    
    def _plot_real_estate_activity(self, df, ax):
        """Plot de l'activité immobilière"""
        ax.bar(df['Year'], df['Home_Sales_Volume'], label='Home Sales', 
              color='#002868', alpha=0.7)
        
        ax.set_title('Real Estate Market Activity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Home Sales Volume', color='#002868')
        ax.tick_params(axis='y', labelcolor='#002868')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Price_per_Sqft'], label='Price per Sqft', 
                linewidth=2, color='#BF0A30')
        ax2.set_ylabel('Price per Sqft ($)', color='#BF0A30')
        ax2.tick_params(axis='y', labelcolor='#BF0A30')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_revenue_expenses(self, df, ax):
        """Plot de l'évolution des recettes et dépenses"""
        ax.plot(df['Year'], df['Total_Revenue'], label='Total Revenue', 
               linewidth=2, color='#002868', alpha=0.8)
        ax.plot(df['Year'], df['Total_Expenses'], label='Total Expenses', 
               linewidth=2, color='#BF0A30', alpha=0.8)
        
        ax.set_title('Revenue and Expenses Evolution (M$)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Amount (M$)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_revenue_structure(self, df, ax):
        """Plot de la structure des recettes (avec énergie)"""
        years = df['Year']
        width = 0.8
        
        bottom = np.zeros(len(years))
        categories = ['Property_Tax_Revenue', 'State_Federal_Funding', 'Business_Tax_Revenue', 'Energy_Revenue', 'Other_Revenue']
        colors = ['#002868', '#BF0A30', '#666666', '#008751', '#FFA300']
        labels = ['Property Tax', 'Govt Funding', 'Business Tax', 'Energy Revenue', 'Other Revenue']
        
        for i, category in enumerate(categories):
            ax.bar(years, df[category], width, label=labels[i], bottom=bottom, color=colors[i])
            bottom += df[category]
        
        ax.set_title('Revenue Structure (M$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amount (M$)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_rental_market(self, df, ax):
        """Plot du marché locatif"""
        ax.plot(df['Year'], df['Average_Rent'], label='Average Rent', 
               linewidth=2, color='#008751', alpha=0.8)
        
        ax.set_title('Rental Market Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Rent ($)', color='#008751')
        ax.tick_params(axis='y', labelcolor='#008751')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Rental_Vacancy_Rate'], label='Vacancy Rate', 
                linewidth=2, color='#BF0A30', alpha=0.8)
        ax2.set_ylabel('Vacancy Rate (%)', color='#BF0A30')
        ax2.tick_params(axis='y', labelcolor='#BF0A30')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_regional_investments(self, df, ax):
        """Plot des investissements régionaux"""
        ax.plot(df['Year'], df['Tech_Investment'], label='Technology', 
               linewidth=2, color='#002868', alpha=0.8)
        ax.plot(df['Year'], df['Energy_Investment'], label='Energy', 
               linewidth=2, color='#BF0A30', alpha=0.8)
        ax.plot(df['Year'], df['Infrastructure_Investment'], label='Infrastructure', 
               linewidth=2, color='#008751', alpha=0.8)
        ax.plot(df['Year'], df['Housing_Development_Investment'], label='Housing', 
               linewidth=2, color='#FFA300', alpha=0.8)
        
        ax.set_title('Regional Investments Distribution (M$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amount (M$)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_demography_income(self, df, ax):
        """Plot de la démographie et des revenus"""
        ax.plot(df['Year'], df['Population']/1000, label='Population', 
               linewidth=2, color='#002868', alpha=0.8)
        
        ax.set_title('Demography and Income Trends', fontsize=12, fontweight='bold')
        ax.set_ylabel('Population (Thousand)', color='#002868')
        ax.tick_params(axis='y', labelcolor='#002868')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Median_Income']/1000, label='Median Income', 
                linewidth=2, color='#BF0A30', alpha=0.8)
        ax2.set_ylabel('Median Income (Thousand $)', color='#BF0A30')
        ax2.tick_params(axis='y', labelcolor='#BF0A30')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_debt_budget(self, df, ax):
        """Plot de la dette et de l'équilibre budgétaire"""
        ax.bar(df['Year'], df['Regional_Debt'], label='Regional Debt (M$)', 
              color='#002868', alpha=0.7)
        
        ax.set_title('Regional Debt and Budget Balance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Debt (M$)', color='#002868')
        ax.tick_params(axis='y', labelcolor='#002868')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Budget_Surplus_Deficit'], label='Budget Balance', 
                linewidth=3, color='#008751')
        ax2.set_ylabel('Budget Balance (M$)', color='#008751')
        ax2.tick_params(axis='y', labelcolor='#008751')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_construction_development(self, df, ax):
        """Plot de la construction et du développement"""
        ax.bar(df['Year'], df['New_Construction_Permits'], label='Construction Permits', 
              color='#FFA300', alpha=0.7)
        
        ax.set_title('Construction and Development Activity', fontsize=12, fontweight='bold')
        ax.set_ylabel('New Construction Permits', color='#FFA300')
        ax.tick_params(axis='y', labelcolor='#FFA300')
        ax.grid(True, alpha=0.3, axis='y')
        
        ax2 = ax.twinx()
        ax2.plot(df['Year'], df['Housing_Development_Investment'], label='Housing Investment', 
                linewidth=2, color='#BF0A30')
        ax2.set_ylabel('Housing Investment (M$)', color='#BF0A30')
        ax2.tick_params(axis='y', labelcolor='#BF0A30')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_sectorial_investments(self, df, ax):
        """Plot des investissements sectoriels"""
        years = df['Year']
        width = 0.8
        
        bottom = np.zeros(len(years))
        categories = ['Tech_Investment', 'Energy_Investment', 'Infrastructure_Investment', 
                     'Housing_Development_Investment', 'Manufacturing_Investment', 'Agricultural_Investment']
        
        colors = ['#002868', '#BF0A30', '#008751', '#FFA300', '#666666', '#8B4513']
        labels = ['Technology', 'Energy', 'Infrastructure', 'Housing', 'Manufacturing', 'Agriculture']
        
        for i, category in enumerate(categories):
            ax.bar(years, df[category], width, label=labels[i], bottom=bottom, color=colors[i])
            bottom += df[category]
        
        ax.set_title('Sectorial Investments Distribution (M$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amount (M$)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _generate_texas_insights(self, df):
        """Génère des insights analytiques adaptés au marché texan"""
        print(f"🤠 TEXAS REAL ESTATE INSIGHTS - {self.region}")
        print("=" * 65)
        
        # 1. Statistiques de base
        print("\n1. 📈 KEY STATISTICS:")
        avg_home_price = df['Median_Home_Price'].mean()
        avg_income = df['Median_Income'].mean()
        avg_rent = df['Average_Rent'].mean()
        price_to_income_ratio = avg_home_price / avg_income
        
        print(f"Average median home price: ${avg_home_price:,.0f}")
        print(f"Average median income: ${avg_income:,.0f}")
        print(f"Average rent: ${avg_rent:.0f}")
        print(f"Price-to-income ratio: {price_to_income_ratio:.1f}")
        
        # 2. Croissance immobilière
        print("\n2. 📊 REAL ESTATE GROWTH:")
        price_growth = ((df['Median_Home_Price'].iloc[-1] / 
                        df['Median_Home_Price'].iloc[0]) - 1) * 100
        population_growth = ((df['Population'].iloc[-1] / 
                             df['Population'].iloc[0]) - 1) * 100
        
        print(f"Home price growth ({self.start_year}-{self.end_year}): {price_growth:.1f}%")
        print(f"Population growth ({self.start_year}-{self.end_year}): {population_growth:.1f}%")
        
        # 3. Accessibilité du logement
        print("\n3. 🏠 HOUSING AFFORDABILITY:")
        current_price = df['Median_Home_Price'].iloc[-1]
        current_income = df['Median_Income'].iloc[-1]
        current_ratio = current_price / current_income
        
        affordability_status = "Critical" if current_ratio > 5 else "Severe" if current_ratio > 4 else "Moderate" if current_ratio > 3 else "Good"
        print(f"Current price-to-income ratio: {current_ratio:.1f} ({affordability_status})")
        
        # 4. Marché locatif
        print("\n4. 🏢 RENTAL MARKET:")
        current_vacancy = df['Rental_Vacancy_Rate'].iloc[-1]
        rent_growth = ((df['Average_Rent'].iloc[-1] / df['Average_Rent'].iloc[0]) - 1) * 100
        
        print(f"Current vacancy rate: {current_vacancy:.1f}%")
        print(f"Rent growth ({self.start_year}-{self.end_year}): {rent_growth:.1f}%")
        
        # 5. Spécificités régionales
        print(f"\n5. 🌟 {self.region.upper()} SPECIFICS:")
        print(f"Region type: {self.config['type']}")
        print(f"Specializations: {', '.join(self.config['specialites'])}")
        print(f"Major cities: {', '.join(self.config['major_cities'])}")
        print(f"Real estate segment: {self.config['segment_immobilier']}")
        
        # 6. Événements marquants du marché texan
        print("\n6. 📅 KEY TEXAS REAL ESTATE EVENTS:")
        print("• 2002-2008: Oil and gas boom driving growth")
        print("• 2008-2009: Mild impact from financial crisis")
        print("• 2010-2014: Shale revolution and energy boom")
        print("• 2015-2016: Oil price collapse affecting energy regions")
        print("• 2015-present: Major tech migration to Texas")
        print("• 2020-2021: COVID-19 pandemic with Texas resilience")
        print("• 2021-present: Massive population growth and development")
        print("• Ongoing: Business-friendly policies attracting companies")
        
        # 7. Recommandations stratégiques
        print("\n7. 💡 STRATEGIC RECOMMENDATIONS:")
        if "énergie" in self.config["specialites"]:
            print("• Diversify beyond oil and gas dependence")
            print("• Invest in renewable energy transition")
        if "technologie" in self.config["specialites"]:
            print("• Continue attracting tech companies and talent")
            print("• Develop innovation districts and tech hubs")
        if "manufacturing" in self.config["specialites"]:
            print("• Support advanced manufacturing development")
            print("• Invest in workforce training programs")
        
        print("• Manage rapid growth with infrastructure investment")
        print("• Maintain housing affordability through supply")
        print("• Invest in transportation and water infrastructure")
        print("• Support small business and entrepreneurship")
        print("• Focus on sustainable development practices")

def main():
    """Fonction principale pour le Texas"""
    # Liste des régions du Texas
    regions = ["Dallas-Fort Worth", "Houston Metro", "Austin Area", "San Antonio", 
               "El Paso Area", "Rio Grande Valley", "West Texas", "Central Texas"]
    
    print("🤠 TEXAS REAL ESTATE ANALYSIS - MAJOR REGIONS (2002-2025)")
    print("=" * 70)
    
    # Demander à l'utilisateur de choisir une région
    print("Available regions:")
    for i, region in enumerate(regions, 1):
        print(f"{i}. {region}")
    
    try:
        choice = int(input("\nSelect the region number to analyze: "))
        if choice < 1 or choice > len(regions):
            raise ValueError
        selected_region = regions[choice-1]
    except (ValueError, IndexError):
        print("Invalid choice. Defaulting to Dallas-Fort Worth.")
        selected_region = "Dallas-Fort Worth"
    
    # Initialiser l'analyseur
    analyzer = TexasRealEstateAnalyzer(selected_region)
    
    # Générer les données
    real_estate_data = analyzer.generate_financial_data()
    
    # Sauvegarder les données
    output_file = f'{selected_region.replace(" ", "_").lower()}_texas_data_2002_2025.csv'
    real_estate_data.to_csv(output_file, index=False)
    print(f"💾 Data saved: {output_file}")
    
    # Aperçu des données
    print("\n👀 Data preview:")
    print(real_estate_data[['Year', 'Population', 'Median_Home_Price', 'Home_Sales_Volume', 'Total_Revenue']].head())
    
    # Créer l'analyse
    print("\n📈 Creating Texas real estate analysis...")
    analyzer.create_financial_analysis(real_estate_data)
    
    print(f"\n✅ Analysis of {selected_region}, Texas completed!")
    print(f"📊 Period: {analyzer.start_year}-{analyzer.end_year}")
    print("🏠 Data: Demographics, real estate market, investments, regional economics")

if __name__ == "__main__":
    main()