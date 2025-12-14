{{- define "my-app.name" -}}
{{.Chart.Name}}
{{- end}}

{{- define "my-app.fullname" -}}
{{ .Release.Name }}-{{ .Chart.Name }}
{{- end }}