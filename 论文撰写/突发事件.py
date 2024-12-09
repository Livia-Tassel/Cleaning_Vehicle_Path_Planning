def main_loop():
    t = 0
    while system_running:
        events = check_events(t)

        if "rain" in events:
            update_dirtiness_rain_mode(G, a, t)
        else:
            if "event" in events or "construction" in events:
                G_prime = update_graph_for_events(G, N_event, E_event)
            else:
                G_prime = G

        for cleaner in C:
            if not cleaner.path:
                target_edge = select_high_dirtiness_edge(G_prime, D)
                lambda_param = choose_lambda_param()
                path = dynamic_A_star(G_prime, cleaner, target_edge, lambda_param, t)
                cleaner.set_path(path)
            else:
                cleaner.move_along_path()
                update_dirtiness(G, cleaner)

        if event_ended:
            restore_graph(G_prime, G)
            increase_dirtiness_post_event(G, E_event, deltaD_event)
            if need_reallocate_vehicles(G, D_threshold):
                reallocate_vehicles(C, G, D, D_threshold)

        t += 1
