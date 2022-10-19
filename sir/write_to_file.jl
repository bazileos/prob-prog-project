function write_to_file_original(choices)
    touch("./sir/parameters.txt")
    file = open("./sir/parameters.txt", "w")
    write(file, "tau: $(choices[:tau])\n")
    write(file, "R0: $(choices[:R0])\n")
    write(file, "rho0: $(choices[:rho0])\n")
    write(file, "rho1: $(choices[:rho1])\n")
    write(file, "rho2: $(choices[:rho2])\n")
    write(file, "switch_to_rho1: $(choices[:switch_to_rho1])\n")
    write(file, "switch_to_rho2: $(choices[:switch_to_rho2])\n")
    close(file)

    touch("./sir/data.txt")
    file = open("./sir/data.txt", "w")
    for t = 1:59
        o_t = choices["o_$t"]
        write(file, "$t\t$o_t\n")
    end
end

function write_to_file_chain(choices, T)
    touch("./sir/parameters.txt")
    file = open("./sir/parameters.txt", "w")
    write(file, "tau: $(choices[:tau])\n")
    write(file, "R0: $(choices[:R0])\n")
    write(file, "rho0: $(choices[:rho0])\n")
    write(file, "rho1: $(choices[:rho1])\n")
    write(file, "rho2: $(choices[:rho2])\n")
    write(file, "switch_to_rho1: $(choices[:switch_to_rho1])\n")
    write(file, "switch_to_rho2: $(choices[:switch_to_rho2])\n")
    close(file)

    touch("./sir/data.txt")
    file = open("./sir/data.txt", "w")
    for t = 1:T
        o_t = choices[:chain => t => :obs]
        write(file, "$t\t$o_t\n")
    end
end