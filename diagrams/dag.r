# Use ggdag to create a causal DAG for the manager training study

# Practice here: https://cbdrh.shinyapps.io/daggle/
# Mark Hanly, Bronwyn K Brew, Anna Austin, Louisa Jorm, Software Application Profile: 
# The daggle app—a tool to support learning and teaching the graphical rules of selecting adjustment variables 
# using directed acyclic graphs, International Journal of Epidemiology, 2023;, dyad038 https://doi.org/10.1093/ije/dyad038

# outcomes: Manager Efficacy, Workload, Turnover Intention, Retention
# treatment: Manager Training
# confounders: Gender, age, tenure, performance rating, organization, job family, region, num. direct reports, baseline survey scores, promotion intensity (driven by organization)
# all confounders -> treatment and outcome
# organization in particular -> promotion intensity -> treatment

# Load required libraries
library(ggdag)
library(tidyverse) 
library(ggplot2)

# Define the DAG structure using dagify()
# Note: ggdag uses ~ syntax where outcome ~ exposure means exposure -> outcome
# Define the DAG structure with collapsed outcomes
manager_training_dag <- dagify(
  # Treatment effect on outcomes (collapsed)
  outcomes ~ manager_training,
  
  # Organization drives promotion intensity (mediator pathway)
  promotion_intensity ~ organization,
  manager_training ~ promotion_intensity,
  
  # Organization direct effects on outcomes
  outcomes ~ organization,
  
  # All confounder groups affect treatment and outcomes
  # Employee Demographics
  manager_training ~ employee_demographics,
  outcomes ~ employee_demographics,
  
  # Performance & Career
  manager_training ~ performance_career,
  outcomes ~ performance_career,
  
  # Role & Structure
  manager_training ~ role_structure,
  outcomes ~ role_structure,
  
  # Baseline Survey Scores
  manager_training ~ baseline_scores,
  outcomes ~ baseline_scores,
  
  # Promotion intensity effects on outcomes
  outcomes ~ promotion_intensity,
  
  # Define node coordinates for clean layout
  coords = list(
    x = c(manager_training = 4, 
          outcomes = 6,
          organization = 1, 
          promotion_intensity = 2.5,
          employee_demographics = 1, 
          performance_career = 1.5,
          role_structure = 2, 
          baseline_scores = 2.5),
    y = c(manager_training = 3.5,
          outcomes = 3.5,
          organization = 4.2, 
          promotion_intensity = 4,
          employee_demographics = 2.5, 
          performance_career = 1.5,
          role_structure = 1, 
          baseline_scores = 3)
  ),
  
  # Define node labels for display
  labels = c(
    manager_training = "Manager\nTraining",
    outcomes = "Outcomes\n(Efficacy, Workload,\nTurnover, Retention)",
    organization = "Organization",
    promotion_intensity = "Promotion\nIntensity",
    employee_demographics = "Employee\nDemographics",
    performance_career = "Performance\n& Career",
    role_structure = "Role &\nStructure",
    baseline_scores = "Baseline\nSurvey Scores"
  ),
  
  # Define exposure and outcome for highlighting
  exposure = "manager_training",
  outcome = "outcomes"
)

# Role mapping for causal inference legend:
# - Treatment: intervention whose effect we estimate
# - Outcome: primary endpoint(s)
# - Confounder: causes both treatment and outcome; may bias estimates if unadjusted
# - Mediator: on causal pathway; adjustment changes interpretation (total vs. direct effect)
role_mapping <- c(
  manager_training = "Treatment (Exposure)",
  outcomes = "Outcome",
  promotion_intensity = "Mediator",
  organization = "Confounder",
  employee_demographics = "Confounder",
  performance_career = "Confounder",
  role_structure = "Confounder",
  baseline_scores = "Confounder"
)
role_colors <- c(
  "Treatment (Exposure)" = "#2E86AB",
  "Outcome" = "#A23B72",
  "Mediator" = "#C73E1D",
  "Confounder" = "#6B7280"
)

# Create the main DAG visualization
dag_plot <- manager_training_dag %>%
  tidy_dagitty() %>%
  mutate(role = recode(name, !!!role_mapping)) %>%
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_edges_link(
    edge_color = "grey60",
    edge_width = 0.8,
    arrow = grid::arrow(length = grid::unit(8, "pt"), type = "closed")
  ) +
  geom_dag_point(aes(color = role), size = 12, alpha = 0.9, stroke = 0) +
  geom_dag_label_repel(
    aes(label = label, fill = role),
    color = "white",
    size = 3.2,
    fontface = "bold",
    box.padding = grid::unit(0.4, "lines"),
    point.padding = grid::unit(2, "lines"),
    force = 2,
    max.iter = 5000,
    show.legend = FALSE
  ) +
  scale_color_manual(
    name = "",
    values = role_colors,
    breaks = names(role_colors)
  ) +
  scale_fill_manual(
    name = "",
    values = role_colors,
    breaks = names(role_colors)
  ) +
  guides(
    color = guide_legend(
      title = "",
      override.aes = list(size = 5),
      order = 1
    ),
    fill = "none"
  ) +
  theme_dag_blank() +
  labs(
    title = "Manager Training Impact: Causal DAG",
  ) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey40"),
    plot.caption = element_text(size = 10, color = "grey50"),
    legend.position = "right",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 9),
    aspect.ratio = 0.5  # Wider panel (height = 50% of width); counters label repulsion stretch
  )

# Display the main plot
print(dag_plot)


# Save plot
ggsave("./diagrams/manager_training_dag.png", dag_plot, width = 12, height = 8, dpi = 300)



