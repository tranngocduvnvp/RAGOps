output "airflow_instance_ip" {
  description = "Public IP of Airflow VM"
  value       = google_compute_instance.airflow_vm.network_interface[0].access_config[0].nat_ip
}

output "service_account_email" {
  value = google_service_account.airflow_sa.email
}
