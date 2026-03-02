SHELL := /bin/bash
BACKEND_DIR := backend

.PHONY: help api worker stack stack-lan deps test migrate migration-new ingest compose-build compose-up compose-down compose-logs compose-ps compose-migrate fly-bootstrap fly-mpg-create fly-mpg-attach fly-deploy fly-status fly-logs fly-health fly-secrets fly-selfhosted-pg-bootstrap fly-pg-sync fly-pg-parity fly-pg-cutover

help:
	@$(MAKE) -C $(BACKEND_DIR) help

api worker stack stack-lan deps test migrate migration-new ingest compose-build compose-up compose-down compose-logs compose-ps compose-migrate fly-bootstrap fly-mpg-create fly-mpg-attach fly-deploy fly-status fly-logs fly-health fly-secrets fly-selfhosted-pg-bootstrap fly-pg-sync fly-pg-parity fly-pg-cutover:
	@$(MAKE) -C $(BACKEND_DIR) $@
