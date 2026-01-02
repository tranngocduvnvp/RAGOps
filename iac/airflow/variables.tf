variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "ragops-481207"
}

variable "region" {
  description = "The region the GCP in"
  default     = "asia-southeast1-b"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "asia-southeast1-b"
}

variable "instance_name" {
  description = "Compute Engine name"
  type        = string
  default     = "airflow-gce"
}
